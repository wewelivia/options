"""Sanity checks for the SABR + Breeden-Litzenberger core."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
from app.core.sabr import sabr_lognormal_vol, calibrate_sabr
from app.core.breeden_litzenberger import extract_rnd, black76_call


def test_roundtrip_lognormal():
    """If the smile is flat (nu=0, beta=1), the RND should match a lognormal
    with that vol, and P(S>F) should be < 0.5 (lognormal is right-skewed in level).
    """
    F, T, vol = 100.0, 0.5, 0.20
    # Build a synthetic flat smile
    strikes = np.linspace(70, 140, 15)
    # beta=1, nu~0 => essentially constant vol
    mkt = np.full_like(strikes, vol)
    params = calibrate_sabr(F, strikes, mkt, T, beta=1.0)
    print(f"Flat-smile calibration RMSE={params.rmse:.2e}, nu={params.nu:.4f}")
    rnd = extract_rnd(params, r=0.0, n_grid=1200)
    st = rnd.stats()
    print("Stats:", {k: round(v,3) for k,v in st.items()})

    # Analytic lognormal check: mean of S_T under Q (r=0) should equal F.
    assert abs(rnd.mean() - F) / F < 0.02, f"mean {rnd.mean()} vs F {F}"
    # PDF integrates to ~1
    assert abs(getattr(np,"trapezoid",getattr(np,"trapz",None))(rnd.pdf, rnd.strikes) - 1.0) < 1e-3
    # Analytic P(S>110) for lognormal forward measure
    from scipy.stats import norm
    d2 = (np.log(F/110) - 0.5*vol**2*T)/(vol*np.sqrt(T))
    analytic = norm.cdf(d2)  # P(S_T > K) = N(d2) for GBM martingale (r=0)
    numeric = rnd.prob_above(110)
    print(f"P(S>110): analytic={analytic:.4f}, numeric={numeric:.4f}")
    assert abs(analytic - numeric) < 0.01, (analytic, numeric)


def test_skewed_smile():
    """A downward-sloping smile (equity skew) should produce a left-skewed RND:
    more mass in the left tail, so P(S < low) is elevated vs lognormal."""
    F, T = 5000.0, 0.25
    strikes = np.array([4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000], float)
    # Typical equity skew: higher vol at low strikes
    mkt = np.array([0.30, 0.27, 0.24, 0.22, 0.20, 0.185, 0.175, 0.17, 0.168])
    params = calibrate_sabr(F, strikes, mkt, T, beta=1.0)
    print(f"Skew calibration RMSE={params.rmse:.4f}, rho={params.rho:.3f}, nu={params.nu:.3f}")
    assert params.rho < 0, "equity skew should give negative rho"
    rnd = extract_rnd(params, r=0.0)
    st = rnd.stats()
    print("Skew stats:", {k: round(v,1) for k,v in st.items()})
    # Left skew => median < mean is NOT guaranteed, but 5th pct should be far below F
    assert rnd.prob_below(4000) > 0.02
    assert abs(getattr(np,"trapezoid",getattr(np,"trapz",None))(rnd.pdf, rnd.strikes) - 1.0) < 1e-3


def test_rates_shifted():
    """Fed funds style: forward ~4.5%, want P(rate > 5%). Uses shifted SABR."""
    F, T, shift = 4.5, 0.5, 3.0  # in percent
    strikes = np.array([3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5], float)
    mkt = np.array([0.55, 0.45, 0.42, 0.40, 0.41, 0.44, 0.52])  # vol smile in these units
    params = calibrate_sabr(F, strikes, mkt, T, beta=0.5, shift=shift)
    print(f"Rates calibration RMSE={params.rmse:.4f}")
    rnd = extract_rnd(params, r=0.0, strike_lo=2.0, strike_hi=8.0, n_grid=1000)
    p = rnd.prob_above(5.0)
    print(f"P(rate > 5%) = {p:.4f}")
    assert 0.0 < p < 1.0
    assert abs(getattr(np,"trapezoid",getattr(np,"trapz",None))(rnd.pdf, rnd.strikes) - 1.0) < 1e-3


if __name__ == "__main__":
    test_roundtrip_lognormal()
    print("--- test_roundtrip_lognormal PASSED ---\n")
    test_skewed_smile()
    print("--- test_skewed_smile PASSED ---\n")
    test_rates_shifted()
    print("--- test_rates_shifted PASSED ---\n")
    print("ALL CORE MATH TESTS PASSED")
