"""
Assemble a provider-agnostic OptionChain.

For the mock provider this simply delegates to build_chain(). For the live
BloombergProvider it orchestrates BDS (chain members) -> BDP (per-option vols,
strikes, expiries) -> group into ExpirySlices and compute forwards.

Kept separate from bloomberg.py so the parsing / grouping logic is unit-testable
without a Terminal.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import numpy as np

from .bloomberg import (ASSET_DEFAULTS, BloombergProvider, MockProvider,
                        OptionChain, OptionQuote, ExpirySlice, act365)


def classify_asset(ticker: str) -> str:
    u = ticker.upper()
    if any(t in u for t in ("FED", "SOFR", " OIS", "RATE")):
        return "RATES"
    if u.endswith("INDEX"):
        return "EQ_INDEX"
    if u.endswith("EQUITY"):
        return "EQUITY"
    if u.endswith("CURNCY"):
        return "FX"
    if u.endswith("COMDTY"):
        return "CMDTY"
    return "EQUITY"


def build_chain(provider, underlying: str, n_expiries: int = 6,
                max_options: int = 1500) -> OptionChain:
    """Dispatch to the correct builder based on provider type."""
    if isinstance(provider, MockProvider):
        return provider.build_chain(underlying, n_expiries=n_expiries)
    if isinstance(provider, BloombergProvider):
        return _build_live(provider, underlying, n_expiries, max_options)
    raise TypeError(f"Unknown provider type: {type(provider)}")


def _build_live(bbg: BloombergProvider, underlying: str, n_expiries: int,
                max_options: int) -> OptionChain:
    """Assemble an OptionChain from a live Bloomberg connection.

    Robust to the usual xbbg quirks: mixed column names, missing IVOL, and
    string expiry/strike fields. Any option row lacking a positive implied vol
    is dropped.
    """
    asset_class = classify_asset(underlying)
    defaults = ASSET_DEFAULTS[asset_class]
    as_of = dt.date.today()

    spot = bbg.spot(underlying)

    # Chain members (calls + puts). Some tickers need overrides; we pull both.
    members = bbg.chain_tickers(underlying, call_put="C")
    if len(members) > max_options:
        members = members[:max_options]

    df = bbg.option_fields(members)
    # df is indexed by ticker with columns per field (xbbg lowercases fields).
    cols = {c.lower(): c for c in df.columns}

    def col(name, *alts):
        for n in (name, *alts):
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_iv = col("ivol_mid", "ivol", "3mo_call_imp_vol")
    c_k = col("opt_strike_px", "strike")
    c_exp = col("opt_expire_dt", "expiry")
    c_pc = col("opt_put_call", "put_call")
    c_undl = col("opt_undl_px")
    c_bid, c_ask, c_last = col("px_bid"), col("px_ask"), col("px_last")
    c_oi, c_vol = col("open_int"), col("px_volume")

    by_exp: dict[dt.date, list[OptionQuote]] = defaultdict(list)
    undl_by_exp: dict[dt.date, list[float]] = defaultdict(list)

    for tk, row in df.iterrows():
        try:
            iv = float(row[c_iv]) if c_iv else None
            if iv is None or not (iv > 0):
                continue
            # xbbg often returns IVOL in percent; normalise to decimal.
            if iv > 3.0:
                iv = iv / 100.0
            K = float(row[c_k])
            exp_raw = row[c_exp]
            exp = _to_date(exp_raw)
            if exp is None or exp <= as_of:
                continue
            pc = str(row[c_pc]).strip().upper()[:1] if c_pc else ("C" if K >= spot else "P")
            pc = "C" if pc in ("C", "1") else "P"
            q = OptionQuote(
                strike=K, expiry=exp, call_put=pc, implied_vol=iv,
                bid=_f(row.get(c_bid)) if c_bid else None,
                ask=_f(row.get(c_ask)) if c_ask else None,
                mid_price=_f(row.get(c_last)) if c_last else None,
                open_interest=_f(row.get(c_oi)) if c_oi else None,
                volume=_f(row.get(c_vol)) if c_vol else None,
            )
            by_exp[exp].append(q)
            if c_undl:
                u = _f(row.get(c_undl))
                if u:
                    undl_by_exp[exp].append(u)
        except Exception:
            continue

    # Keep the nearest n_expiries with a reasonable number of strikes.
    exps = sorted(e for e, qs in by_exp.items() if len(qs) >= 5)[:n_expiries]
    slices = []
    for e in exps:
        qs = by_exp[e]
        F = float(np.median(undl_by_exp[e])) if undl_by_exp.get(e) else spot
        slices.append(ExpirySlice(expiry=e, forward=F, T=act365(as_of, e), quotes=qs))

    return OptionChain(underlying=underlying, asset_class=asset_class, spot=float(spot),
                       as_of=as_of, expiries=slices, source="bloomberg",
                       shift=float(defaults["shift"]))


def _to_date(v):
    if isinstance(v, dt.date):
        return v
    if isinstance(v, dt.datetime):
        return v.date()
    try:
        import pandas as pd
        return pd.to_datetime(v).date()
    except Exception:
        return None


def _f(v):
    try:
        if v is None:
            return None
        f = float(v)
        return None if (f != f) else f
    except Exception:
        return None
