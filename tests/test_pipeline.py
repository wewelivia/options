"""End-to-end pipeline test using the mock provider."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import datetime as dt
from app.data.bloomberg import MockProvider, ASSET_DEFAULTS
from app.data.chain_builder import build_chain
from app.core.sabr import calibrate_sabr
from app.core.breeden_litzenberger import extract_rnd
from app.core.event_parser import parse_condition, compute_probability


def run(underlying, condition, force_percent=None):
    prov = MockProvider()
    chain = build_chain(prov, underlying)
    spec = parse_condition(condition, force_percent=force_percent)
    sl = chain.nearest_expiry(spec.target_date)
    strikes, vols = sl.smile()
    beta = ASSET_DEFAULTS[chain.asset_class]["beta"]
    params = calibrate_sabr(sl.forward, strikes, vols, sl.T, beta=beta, shift=chain.shift)
    # Widen the RND grid well beyond quoted strikes so the tails are captured.
    lo = max(strikes.min()*0.3, (chain.spot - 4*abs(chain.spot)) if chain.asset_class=='RATES' else strikes.min()*0.3)
    if chain.asset_class == 'RATES':
        lo, hi = sl.forward - 4.0, sl.forward + 4.0
    else:
        lo, hi = strikes.min()*0.5, strikes.max()*1.6
    rnd = extract_rnd(params, r=0.0, strike_lo=lo, strike_hi=hi, n_grid=1200)
    res = compute_probability(rnd, spec)
    print(f"\n{underlying}: {condition!r}")
    print(f"  asset_class={chain.asset_class} forward={sl.forward} T={sl.T:.3f} expiry={sl.expiry}")
    print(f"  SABR: alpha={params.alpha:.4f} beta={params.beta} rho={params.rho:.3f} nu={params.nu:.3f} rmse={params.rmse:.4f}")
    print(f"  ==> P({res['condition']}) = {res['probability']*100:.1f}%")
    assert 0 <= res["probability"] <= 1
    stats = rnd.stats()
    print(f"  RND median={stats['median']:.2f} p05={stats['p05']:.2f} p95={stats['p95']:.2f}")
    return res


if __name__ == "__main__":
    run("SPX Index", "SPX above 6000 by December")
    run("SPX Index", "SPX below 5000 by year end")
    run("AAPL US Equity", "AAPL between 200 and 240 by Jan 2027")
    run("EURUSD Curncy", "EURUSD above 1.12 by December")
    run("XAU Curncy", "Gold above 2500 by year end")
    run("FEDFUNDS", "Fed funds rate above 5% by December", force_percent=True)
    run("FEDFUNDS", "Fed funds below 4% by December", force_percent=True)
    print("\nALL PIPELINE TESTS PASSED")
