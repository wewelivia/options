"""
Orchestration service: ties data -> SABR -> Breeden-Litzenberger -> event
probability into single callable used by the API layer.
"""
from __future__ import annotations

import datetime as dt
import math
from functools import lru_cache

import numpy as np

from ..data.bloomberg import (ASSET_DEFAULTS, BloombergProvider, MockProvider,
                              get_provider)
from ..data.chain_builder import build_chain, classify_asset
from .sabr import calibrate_sabr
from .breeden_litzenberger import extract_rnd
from .event_parser import parse_condition, compute_probability


# Cache chains briefly so repeated requests on the same underlying are fast.
_CHAIN_CACHE: dict[str, tuple[float, object]] = {}
_CHAIN_TTL = 60.0  # seconds


def _get_chain(underlying: str, prefer_live: bool = True):
    import time
    now = time.time()
    hit = _CHAIN_CACHE.get(underlying)
    if hit and (now - hit[0] < _CHAIN_TTL):
        return hit[1]
    provider = get_provider(prefer_live=prefer_live)
    chain = build_chain(provider, underlying)
    _CHAIN_CACHE[underlying] = (now, chain)
    return chain


def _shape(obj) -> str:
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            cols = list(obj.columns)[:6]
            return (f"DataFrame shape={obj.shape} "
                    f"cols_type={type(obj.columns).__name__} "
                    f"cols_sample={cols}")
        if isinstance(obj, pd.Series):
            return f"Series len={len(obj)} index_sample={list(obj.index)[:6]}"
    except Exception:
        pass
    return f"{type(obj).__name__}"


def diagnose(underlying: str) -> dict:
    """Probe the live Bloomberg path one call at a time, capturing the raw
    return shape and any error per step. Never raises -- returns a report the
    UI/terminal can show so we can pinpoint exactly where a chain build fails.
    """
    report: dict = {"underlying": underlying,
                    "classified_as": classify_asset(underlying),
                    "steps": []}

    prov = get_provider(prefer_live=True)
    is_live = isinstance(prov, BloombergProvider)
    report["provider"] = "bloomberg" if is_live else "mock"
    if not is_live:
        report["note"] = ("No live Terminal detected (xbbg/blpapi not importable or "
                          "not connected). The app is using synthetic surfaces.")
        # Still show the mock builds cleanly.
        try:
            chain = build_chain(prov, underlying)
            report["steps"].append({"step": "mock_build", "ok": True,
                                    "expiries": len(chain.expiries),
                                    "asset_class": chain.asset_class})
        except Exception as e:
            report["steps"].append({"step": "mock_build", "ok": False, "error": repr(e)})
        return report

    # ---- live probes ----
    def probe(name, fn):
        import traceback as _tb
        entry = {"step": name}
        try:
            out = fn()
            entry["ok"] = True
            entry["result"] = out
        except Exception as e:
            entry["ok"] = False
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = _tb.format_exc().splitlines()[-4:]
        report["steps"].append(entry)
        return entry.get("result")

    # 1) raw spot frame shape
    def _raw_spot():
        df = prov._xbbg.bdp(tickers=underlying, flds=["PX_LAST"])
        return _shape(df)
    probe("bdp_px_last_shape", _raw_spot)

    # 2) parsed spot
    spot = probe("spot_value", lambda: prov.spot(underlying))

    # 3) raw OPT_CHAIN shape
    def _raw_chain():
        df = prov._xbbg.bds(tickers=underlying, flds="OPT_CHAIN")
        return _shape(df)
    probe("bds_opt_chain_shape", _raw_chain)

    # 4) parsed chain members (count + sample)
    def _members():
        m = prov.chain_tickers(underlying)
        return {"count": len(m), "sample": m[:5]}
    members_res = probe("chain_members", _members)

    # 5) option fields for a small sample
    def _fields():
        m = prov.chain_tickers(underlying)[:20]
        df = prov.option_fields(m)
        return {"shape": _shape(df), "columns": list(getattr(df, "columns", []))[:12]}
    probe("option_fields_sample", _fields)

    # 6) full build
    def _build():
        chain = build_chain(prov, underlying)
        return {"expiries": len(chain.expiries),
                "asset_class": chain.asset_class,
                "first_expiry": chain.expiries[0].expiry.isoformat() if chain.expiries else None,
                "n_strikes_first": (len({q.strike for q in chain.expiries[0].quotes})
                                    if chain.expiries else 0)}
    probe("full_build_chain", _build)

    return report


def get_chain_info(underlying: str, prefer_live: bool = True) -> dict:
    chain = _get_chain(underlying, prefer_live)
    return {
        "underlying": chain.underlying,
        "asset_class": chain.asset_class,
        "spot": chain.spot,
        "as_of": chain.as_of.isoformat(),
        "source": chain.source,
        "shift": chain.shift,
        "expiries": [
            {"expiry": e.expiry.isoformat(), "T": e.T, "forward": e.forward,
             "n_strikes": len({q.strike for q in e.quotes})}
            for e in chain.expiries
        ],
    }


def _grid_bounds(chain, sl, strikes):
    if chain.asset_class == "RATES":
        span = max(4.0, 6.0 * sl.forward * math.sqrt(max(sl.T, 0.05)) / 4.0)
        return sl.forward - min(sl.forward + chain.shift - 1e-6, span), sl.forward + span
    return float(strikes.min() * 0.4), float(strikes.max() * 1.7)


def _fmt_odds(p: float) -> str:
    if p <= 0:
        return "~0 (effectively impossible)"
    if p >= 1:
        return "~1 (effectively certain)"
    # implied "X to 1 against"
    against = (1 - p) / p
    if against >= 1:
        return f"{against:.1f} to 1 against"
    return f"{1/against:.1f} to 1 on"


def compute_distribution(underlying: str, condition: str,
                         beta: float | None = None, r: float = 0.0,
                         force_percent: bool | None = None,
                         expiry: str | None = None,
                         prefer_live: bool = True,
                         n_out: int = 250) -> dict:
    """Full pipeline: returns everything the frontend needs to render."""
    chain = _get_chain(underlying, prefer_live)
    if not chain.expiries:
        raise ValueError(f"No option expiries available for {underlying!r}")

    # Determine percent semantics: rates are percent by default.
    fp = force_percent
    if fp is None:
        fp = chain.asset_class == "RATES"

    spec = parse_condition(condition, force_percent=fp)

    # Choose expiry
    if expiry:
        target = dt.date.fromisoformat(expiry)
        sl = min(chain.expiries, key=lambda e: abs((e.expiry - target).days))
    else:
        sl = chain.nearest_expiry(spec.target_date)

    strikes, mvols = sl.smile()
    if len(strikes) < 3:
        raise ValueError(f"Too few strikes ({len(strikes)}) to calibrate at expiry {sl.expiry}")

    b = beta if beta is not None else ASSET_DEFAULTS[chain.asset_class]["beta"]
    params = calibrate_sabr(sl.forward, strikes, mvols, sl.T, beta=b, shift=chain.shift)

    lo, hi = _grid_bounds(chain, sl, strikes)
    rnd = extract_rnd(params, r=r, strike_lo=lo, strike_hi=hi, n_grid=1200)

    ev = compute_probability(rnd, spec)

    # Fitted vols at market strikes for the smile overlay
    fitted = params.vol(strikes)
    smile = [{"strike": float(k), "market_vol": float(mv), "fitted_vol": float(fv)}
             for k, mv, fv in zip(strikes, mvols, fitted)]

    # Downsample RND arrays for transport
    idx = np.linspace(0, len(rnd.strikes) - 1, min(n_out, len(rnd.strikes))).astype(int)
    grid = rnd.strikes[idx].tolist()
    pdf = rnd.pdf[idx].tolist()
    cdf = rnd.cdf[idx].tolist()

    p = ev["probability"]
    return {
        "underlying": chain.underlying,
        "asset_class": chain.asset_class,
        "source": chain.source,
        "expiry": sl.expiry.isoformat(),
        "T": sl.T,
        "forward": sl.forward,
        "is_percent": spec.is_percent,
        "sabr": params.as_dict(),
        "smile": smile,
        "grid": grid,
        "pdf": pdf,
        "cdf": cdf,
        "stats": rnd.stats(),
        "probability": p,
        "condition": ev["condition"],
        "direction": ev["direction"],
        "threshold": ev["threshold"],
        "threshold_hi": ev["threshold_hi"],
        "target_date": ev["target_date"],
        "complement": 1.0 - p,
        "odds": _fmt_odds(p),
    }
