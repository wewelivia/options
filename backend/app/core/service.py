"""
Orchestration service: ties data -> SABR -> Breeden-Litzenberger -> event
probability into single callable used by the API layer.
"""
from __future__ import annotations

import datetime as dt
import math
from functools import lru_cache

import numpy as np

from ..data.bloomberg import ASSET_DEFAULTS, MockProvider, get_provider
from ..data.chain_builder import build_chain
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
