"""
Bloomberg data layer.

Design goal (per user): use **xbbg** for the standard pulls
  - BDP  -> spot / reference data (last price, futures ref, days-to-expiry)
  - BDS  -> chain members (option chain via OPT_CHAIN / bulk fields)
  - BDH  -> historical implied vols where needed
and drop to **blpapi** directly only for things xbbg does not expose cleanly.

The layer degrades gracefully:
  live xbbg  ->  (fallback) blpapi  ->  (fallback) MockProvider
so the FastAPI app boots and is fully explorable on any machine, and switches
to live Terminal data automatically when run where blpapi/xbbg can connect.

All providers return a common `OptionChain` structure so the calibration /
RND layer is provider-agnostic.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------------
# Common data structures
# ----------------------------------------------------------------------------
@dataclass
class OptionQuote:
    strike: float
    expiry: dt.date
    call_put: str            # 'C' or 'P'
    implied_vol: float       # decimal, e.g. 0.20
    mid_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    open_interest: Optional[float] = None
    volume: Optional[float] = None


@dataclass
class ExpirySlice:
    expiry: dt.date
    forward: float
    T: float                 # year fraction to expiry (ACT/365)
    quotes: list[OptionQuote] = field(default_factory=list)

    def smile(self, call_put: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (strikes, implied_vols) sorted by strike.

        If call_put is None we prefer OTM options on each side of the forward
        (calls above F, puts below F) since those carry the cleanest vols, and
        merge them into a single smile.
        """
        qs = self.quotes
        if call_put in ("C", "P"):
            sel = [q for q in qs if q.call_put == call_put]
        else:
            sel = [q for q in qs
                   if (q.call_put == "C" and q.strike >= self.forward)
                   or (q.call_put == "P" and q.strike < self.forward)]
        sel = [q for q in sel if q.implied_vol and q.implied_vol > 0]
        sel.sort(key=lambda q: q.strike)
        # De-duplicate strikes (keep first)
        seen, ks, vs = set(), [], []
        for q in sel:
            if q.strike in seen:
                continue
            seen.add(q.strike)
            ks.append(q.strike)
            vs.append(q.implied_vol)
        return np.array(ks, float), np.array(vs, float)


@dataclass
class OptionChain:
    underlying: str
    asset_class: str          # 'RATES' | 'EQ_INDEX' | 'EQUITY' | 'FX' | 'CMDTY'
    spot: float
    as_of: dt.date
    expiries: list[ExpirySlice] = field(default_factory=list)
    source: str = "mock"
    shift: float = 0.0        # displacement for shifted-SABR (rates)

    def expiry_dates(self) -> list[dt.date]:
        return [e.expiry for e in self.expiries]

    def nearest_expiry(self, target: dt.date) -> ExpirySlice:
        return min(self.expiries, key=lambda e: abs((e.expiry - target).days))


# ----------------------------------------------------------------------------
# Asset-class conventions
# ----------------------------------------------------------------------------
ASSET_DEFAULTS = {
    # beta and shift conventions per asset class for SABR
    "RATES":    {"beta": 0.5, "shift": 3.0},   # percent units; shift 3% displacement
    "EQ_INDEX": {"beta": 1.0, "shift": 0.0},
    "EQUITY":   {"beta": 1.0, "shift": 0.0},
    "FX":       {"beta": 0.5, "shift": 0.0},
    "CMDTY":    {"beta": 0.5, "shift": 0.0},
}


def act365(as_of: dt.date, expiry: dt.date) -> float:
    return max((expiry - as_of).days, 1) / 365.0


# ----------------------------------------------------------------------------
# xbbg return-shape helpers
#
# xbbg's bdp/bds do not have a single stable output shape across versions:
#  - columns may be a flat Index of field names, OR a MultiIndex of
#    (ticker, field) tuples;
#  - a single-ticker bdp may come back as a 1xN frame or, occasionally, a
#    Series; and a missing field yields an empty frame.
# These helpers normalise all of that so the provider never assumes a shape
# and never calls .lower() on a tuple column label (the source of the
# "'DataFrame' object has no attribute 'iloc'" style breakages).
# ----------------------------------------------------------------------------
def _to_pandas(obj):
    """Coerce whatever xbbg returns into a native pandas object.

    Some environments have xbbg (or its deps) return a **narwhals** wrapper
    (`narwhals.stable.v1.DataFrame`) or a polars frame rather than a pandas
    DataFrame. Those look similar (they have .shape) but lack .empty / pandas
    .iloc semantics, which breaks downstream parsing. We unwrap to real pandas
    once, at the boundary, so nothing else has to care.
    """
    if obj is None:
        return None
    try:
        import pandas as pd
    except Exception:
        return obj
    # Already native pandas?
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj
    # narwhals: has .to_native() that returns the underlying frame
    to_native = getattr(obj, "to_native", None)
    if callable(to_native):
        try:
            obj = to_native()
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj
        except Exception:
            pass
    # narwhals also exposes .to_pandas() in some versions
    to_pandas = getattr(obj, "to_pandas", None)
    if callable(to_pandas):
        try:
            return to_pandas()
        except Exception:
            pass
    # polars frame
    mod = type(obj).__module__ or ""
    if mod.startswith("polars"):
        try:
            return obj.to_pandas()
        except Exception:
            pass
    # last resort: try constructing a DataFrame from it
    try:
        return pd.DataFrame(obj)
    except Exception:
        return obj


def _flat_columns(df) -> list[str]:
    """Return column labels as lowercase strings, joining MultiIndex tuples."""
    out = []
    for c in df.columns:
        if isinstance(c, tuple):
            out.append("|".join(str(x) for x in c).lower())
        else:
            out.append(str(c).lower())
    return out


def _coerce_float(v):
    """Return v as a float if it is numeric (or a numeric-looking string),
    else None. Ticker labels like 'SPX Index' return None."""
    if v is None or (isinstance(v, float) and v != v):  # None / NaN
        return None
    if isinstance(v, (int, float)):
        return float(v)
    # numpy scalar
    try:
        import numpy as np
        if isinstance(v, np.generic):
            f = float(v)
            return f if f == f else None
    except Exception:
        pass
    # numeric-looking string (but NOT a ticker like 'SPX Index')
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        try:
            return float(s)
        except (ValueError, TypeError):
            return None
    return None


def _first_scalar(df):
    """Extract the first non-null **numeric** scalar from a bdp result.

    After narwhals->pandas coercion the ticker (formerly the row index) can
    appear as a data cell, so we must skip non-numeric values (e.g. the string
    'SPX Index') and return the first genuine number.
    """
    try:
        import pandas as pd
    except Exception:
        pd = None
    if df is None:
        return None
    # Series
    if pd is not None and isinstance(df, pd.Series):
        for v in df.tolist():
            f = _coerce_float(v)
            if f is not None:
                return f
        return None
    # DataFrame
    if hasattr(df, "empty"):
        if df.empty:
            return None
        # Prefer a PX_LAST-like column if present, else scan all cells.
        try:
            cols = _flat_columns(df)
            for want in ("px_last", "last_price", "px_mid", "px_bid"):
                if want in cols:
                    col = df.iloc[:, cols.index(want)]
                    for v in col.tolist():
                        f = _coerce_float(v)
                        if f is not None:
                            return f
        except Exception:
            pass
        for v in df.to_numpy().ravel():
            f = _coerce_float(v)
            if f is not None:
                return f
        return None
    return None


def _flatten_bdp(df):
    """Return a bdp frame with flat, lowercase field-name columns indexed by
    ticker. Handles both flat and (ticker, field) MultiIndex column layouts,
    and a single-row Series."""
    try:
        import pandas as pd
    except Exception:
        return df
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    # If narwhals->pandas coercion reset the ticker index into a column, put it
    # back as the row index so downstream lookups by ticker keep working.
    if not isinstance(df.columns, pd.MultiIndex):
        lc = [str(c).lower() for c in df.columns]
        for cand in ("ticker", "tickers", "security", "index", "level_0"):
            if cand in lc:
                col = df.columns[lc.index(cand)]
                try:
                    df = df.set_index(col)
                except Exception:
                    pass
                break
    # LONG -> WIDE: some xbbg/narwhals stacks return a melted frame with just
    # ('field','value') columns and the ticker on the row index -- i.e. one row
    # per (ticker, field). Pivot it so each field becomes its own column,
    # indexed by ticker, which is what the chain builder expects.
    lc_now = [str(c).lower() for c in df.columns]
    if not isinstance(df.columns, pd.MultiIndex) and set(lc_now) >= {"field", "value"} \
            and len(df.columns) <= 3:
        try:
            fcol = df.columns[lc_now.index("field")]
            vcol = df.columns[lc_now.index("value")]
            tmp = df[[fcol, vcol]].copy()
            tmp["__tk__"] = df.index
            tmp[fcol] = tmp[fcol].astype(str).str.lower()
            wide = tmp.pivot_table(index="__tk__", columns=fcol, values=vcol,
                                   aggfunc="first")
            wide.index.name = None
            wide.columns = [str(c).lower() for c in wide.columns]
            # Coerce numeric-looking strings to floats where possible (per
            # column; leave genuinely non-numeric fields like opt_put_call as-is).
            for c in wide.columns:
                converted = pd.to_numeric(wide[c], errors="coerce")
                # Keep the numeric version only if it didn't wipe real data
                # (i.e. the column was mostly numeric to begin with).
                orig_nonnull = wide[c].notna().sum()
                if orig_nonnull == 0 or converted.notna().sum() >= 0.5 * orig_nonnull:
                    wide[c] = converted
            return wide
        except Exception:
            pass
    if isinstance(df.columns, pd.MultiIndex):
        # Collapse (ticker, field) -> field, keeping ticker on the row index.
        # xbbg already indexes rows by ticker for bdp, so just take the last
        # level (the field name) as the column label.
        df = df.copy()
        df.columns = [str(c[-1]).lower() for c in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
    return df


# ----------------------------------------------------------------------------
# xbbg / blpapi provider
# ----------------------------------------------------------------------------
class BloombergProvider:
    """Live provider backed by xbbg (preferred) with blpapi as a low-level
    fallback for request types xbbg does not surface cleanly.

    This class is import-safe: it only imports xbbg/blpapi at call time so the
    module loads on machines without a Terminal.
    """

    def __init__(self) -> None:
        self._xbbg = None
        self._blpapi_ok = False
        try:
            import xbbg  # noqa: F401
            from xbbg import blp
            self._xbbg = blp
        except Exception:
            self._xbbg = None
        try:
            import blpapi  # noqa: F401
            self._blpapi_ok = True
        except Exception:
            self._blpapi_ok = False

    @property
    def available(self) -> bool:
        return self._xbbg is not None or self._blpapi_ok

    # ---- reference / spot (BDP) --------------------------------------------
    def spot(self, ticker: str) -> float:
        """Last price via BDP.

        xbbg's bdp() returns a DataFrame indexed by ticker with (lowercased)
        field columns, but the exact shape varies by version and can be empty
        if the field is unavailable. Extract the single scalar defensively
        rather than assuming an (0, 0) position.
        """
        blp = self._xbbg
        df = _to_pandas(blp.bdp(tickers=ticker, flds=["PX_LAST"]))
        val = _first_scalar(df)
        if val is None:
            raise ValueError(f"No PX_LAST returned for {ticker!r} (check the ticker / entitlements)")
        return float(val)

    # ---- chain members (BDS) -----------------------------------------------
    def chain_tickers(self, ticker: str, call_put: str = "C") -> list[str]:
        """Pull option chain member tickers via BDS on OPT_CHAIN.

        Bloomberg field 'OPT_CHAIN' returns a bulk table. xbbg represents this
        as a DataFrame whose columns may be a flat Index OR a MultiIndex
        (ticker, field) -- so column labels can be tuples, not strings. We
        flatten the columns to strings first, then pick the security-description
        column, and finally fall back to the first column by position.
        """
        blp = self._xbbg
        df = _to_pandas(blp.bds(tickers=ticker, flds="OPT_CHAIN"))
        if df is None or getattr(df, "empty", True):
            raise ValueError(f"OPT_CHAIN returned no members for {ticker!r}")

        # Flatten (possibly MultiIndex / tuple) column labels to lowercase strings.
        flat = _flat_columns(df)
        parent = str(ticker).strip().lower()

        def _clean(idx):
            vals = [str(v).strip() for v in df.iloc[:, idx].tolist()
                    if v is not None and str(v).strip()]
            # Drop the parent ticker (index leakage / self-reference).
            return [v for v in vals if v.lower() != parent]

        def _optionlike(vals):
            # A real member ticker has letters, some length, and is not just a
            # number or a bare field label.
            return [v for v in vals
                    if any(c.isalpha() for c in v) and len(v) > 3
                    and v.lower() not in ("opt_chain", "field", "ticker",
                                          "security description")]

        # 1) Prefer an explicitly named member/description column, in priority
        #    order. Note 'security description' holds the actual members; a bare
        #    'ticker' column here is usually the *parent* repeated, so it is LAST.
        preferred = ["security description", "security_description",
                     "description", "security", "member", "members",
                     "option", "ticker"]
        ordered_idxs: list[int] = []
        for name in preferred:
            for i, label in enumerate(flat):
                if label == name and i not in ordered_idxs:
                    ordered_idxs.append(i)
        # 2) Then any remaining columns, so we still find members if labels differ.
        ordered_idxs += [i for i in range(df.shape[1]) if i not in ordered_idxs]

        best: list[str] = []
        for idx in ordered_idxs:
            members = _optionlike(_clean(idx))
            if len(members) > len(best):
                best = members
            # A labelled description/security column that yields any members is
            # authoritative -- stop early.
            if flat[idx] in ("security description", "security_description",
                             "description", "security") and members:
                best = members
                break

        if not best:
            # Surface a rich preview so we can see the real cell contents.
            preview = {}
            try:
                for i, label in enumerate(flat[:6]):
                    preview[label] = [str(v) for v in df.iloc[:, i].tolist()[:4]]
            except Exception:
                pass
            raise ValueError(
                f"OPT_CHAIN for {ticker!r} returned rows but no member tickers "
                f"could be parsed (columns={flat}, preview={preview})")
        return best

    # ---- per-option implied vol + price (BDP, bulk) ------------------------
    def option_fields(self, tickers: list[str]):
        """Bulk BDP for the option fields, returned as a *flat-column* frame
        indexed by ticker so chain_builder can address columns by lowercase
        field name regardless of xbbg's MultiIndex convention."""
        blp = self._xbbg
        flds = ["PX_BID", "PX_ASK", "PX_LAST", "IVOL_MID",
                "OPT_STRIKE_PX", "OPT_EXPIRE_DT", "OPT_PUT_CALL",
                "OPEN_INT", "PX_VOLUME", "OPT_UNDL_PX"]
        df = _to_pandas(blp.bdp(tickers=tickers, flds=flds))
        return _flatten_bdp(df)

    # ---- historical vol (BDH) ----------------------------------------------
    def hist_vol(self, ticker: str, start: dt.date, end: dt.date, fld: str = "3MO_IMPVOL_100.0%MNY_DF"):
        blp = self._xbbg
        return blp.bdh(tickers=ticker, flds=fld, start_date=start, end_date=end)

    # NOTE: build_chain() that assembles OptionChain from the above lives in
    # chain_builder.py so the parsing logic is testable independently of the
    # live connection. When no live connection is present the app uses
    # MockProvider below.


# ----------------------------------------------------------------------------
# Mock provider -- realistic synthetic surfaces so the app is fully usable
# without a Terminal. Produces asset-class-appropriate skew/smile shapes.
# ----------------------------------------------------------------------------
class MockProvider:
    """Generates plausible option chains with realistic smiles per asset class.

    Not random noise: uses a seeded SABR-like shape so calibration recovers
    sensible parameters and the demo is deterministic.
    """

    # Reference spots / forwards for well-known demo underlyings.
    PRESETS = {
        "SPX Index":     ("EQ_INDEX", 5500.0),
        "NDX Index":     ("EQ_INDEX", 19500.0),
        "UKX Index":     ("EQ_INDEX", 8200.0),
        "AAPL US Equity":("EQUITY", 210.0),
        "NVDA US Equity":("EQUITY", 128.0),
        "TSLA US Equity":("EQUITY", 245.0),
        "EURUSD Curncy": ("FX", 1.08),
        "GBPUSD Curncy": ("FX", 1.27),
        "XAU Curncy":    ("CMDTY", 2350.0),
        "CL1 Comdty":    ("CMDTY", 78.0),
        # Rates: express the underlying as the RATE in percent (e.g. implied
        # policy rate). Fed funds / SOFR style.
        "FEDFUNDS":      ("RATES", 4.50),
        "SOFR":          ("RATES", 4.60),
    }

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def resolve(self, underlying: str) -> tuple[str, float]:
        key = underlying.strip()
        if key in self.PRESETS:
            return self.PRESETS[key]
        # Heuristic classification for unknown tickers.
        u = key.upper()
        if u.endswith("INDEX"):
            return ("EQ_INDEX", 5000.0)
        if u.endswith("EQUITY"):
            return ("EQUITY", 100.0)
        if u.endswith("CURNCY"):
            return ("FX", 1.0)
        if u.endswith("COMDTY"):
            return ("CMDTY", 80.0)
        if any(t in u for t in ("FED", "SOFR", "RATE", "OIS")):
            return ("RATES", 4.5)
        return ("EQUITY", 100.0)

    def _smile_vol(self, asset_class: str, F: float, K: np.ndarray, T: float) -> np.ndarray:
        """A stylised, *calibratable* smile in log-moneyness.

        vol(m) = atm + skew*m + conv*m^2, with m = log(K/F) for price
        underlyings and m = (K-F) for rates (percent units). Coefficients are
        deliberately mild so the resulting smile is arbitrage-consistent and
        SABR calibrates cleanly (this is synthetic demo data, not a stress of
        the fitter).
        """
        if asset_class == "RATES":
            m = (K - F)                     # absolute rate difference, percent
            atm, skew, conv = 0.40, 0.010, 0.020
        else:
            m = np.log(K / F)               # log-moneyness
            if asset_class == "EQ_INDEX":
                atm, skew, conv = 0.16, -0.12, 0.30   # negative skew, mild convexity
            elif asset_class == "EQUITY":
                atm, skew, conv = 0.30, -0.10, 0.40
            elif asset_class == "FX":
                atm, skew, conv = 0.09, 0.02, 0.25
            else:  # CMDTY
                atm, skew, conv = 0.26, 0.05, 0.30
        vol = atm + skew * m + conv * m * m
        # term-structure: vols rise slightly with sqrt(T)
        vol = vol * (0.9 + 0.2 * math.sqrt(max(T, 0.01)))
        return np.clip(vol, 0.01, 5.0)

    def build_chain(self, underlying: str, n_expiries: int = 6) -> OptionChain:
        asset_class, F0 = self.resolve(underlying)
        as_of = dt.date.today()
        defaults = ASSET_DEFAULTS[asset_class]
        shift = defaults["shift"]

        # Strike grid per asset class
        if asset_class == "RATES":
            strikes = np.round(np.arange(F0 - 2.0, F0 + 2.01, 0.125), 3)
        elif asset_class == "FX":
            strikes = np.round(F0 * np.linspace(0.85, 1.15, 25), 4)
        else:
            strikes = np.round(F0 * np.linspace(0.6, 1.5, 31), 2)

        expiries = []
        base = as_of
        for i in range(1, n_expiries + 1):
            exp = base + dt.timedelta(days=int(30.4 * i) + 15)
            T = act365(as_of, exp)
            # simple forward: flat (no carry) for the mock
            F = F0
            vols = self._smile_vol(asset_class, F, strikes, T)
            quotes = []
            for K, v in zip(strikes, vols):
                cp = "C" if K >= F else "P"
                quotes.append(OptionQuote(strike=float(K), expiry=exp, call_put=cp,
                                          implied_vol=float(v),
                                          open_interest=float(self.rng.integers(50, 5000)),
                                          volume=float(self.rng.integers(0, 2000))))
            expiries.append(ExpirySlice(expiry=exp, forward=float(F), T=T, quotes=quotes))

        return OptionChain(underlying=underlying, asset_class=asset_class, spot=float(F0),
                           as_of=as_of, expiries=expiries, source="mock", shift=float(shift))


# ----------------------------------------------------------------------------
# Provider selector
# ----------------------------------------------------------------------------
def get_provider(prefer_live: bool = True):
    """Return a live BloombergProvider if a Terminal connection is available,
    else a MockProvider. The FastAPI layer calls build_chain() on whatever is
    returned (both expose a compatible surface via chain_builder)."""
    if prefer_live:
        bbg = BloombergProvider()
        if bbg.available:
            return bbg
    return MockProvider()
