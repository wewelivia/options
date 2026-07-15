"""
Breeden-Litzenberger risk-neutral density (RND) extraction.

Given a calibrated SABR smile, we:
  1. Build a dense strike grid.
  2. Convert SABR implied vols -> Black call prices at each strike.
  3. Apply the Breeden-Litzenberger identity:
         q(K) = e^{rT} * d^2 C / dK^2
     to recover the risk-neutral PDF, computed on the fitted (smooth) price
     curve rather than raw market quotes -- this is why we fit SABR first.
  4. Integrate to get the CDF and answer event-probability questions.

Breeden, Litzenberger (1978), "Prices of State-Contingent Claims Implicit in
Option Prices", Journal of Business.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .sabr import SABRParams

# numpy>=2 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ----------------------------------------------------------------------------
# Black-76 pricing (options on forwards/futures)
# ----------------------------------------------------------------------------
def black76_call(F: float, K: np.ndarray, T: float, sigma: np.ndarray, r: float,
                 shift: float = 0.0) -> np.ndarray:
    """Black-76 call price given forward F, discounting at rate r over T.

    Supports a displacement `shift` consistent with shifted-SABR so that low /
    negative-rate underlyings price correctly.
    """
    F = F + shift
    K = np.asarray(K, float) + shift
    sigma = np.asarray(sigma, float)
    T = max(T, 1e-8)
    sqrtT = np.sqrt(T)
    # Guard tiny vols
    sigma = np.clip(sigma, 1e-8, None)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = np.exp(-r * T)
    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))


@dataclass
class RiskNeutralDensity:
    strikes: np.ndarray          # dense strike grid (unshifted / real-world units)
    pdf: np.ndarray              # risk-neutral pdf q(K)
    cdf: np.ndarray              # risk-neutral cdf
    call_prices: np.ndarray      # fitted call prices used
    fitted_vols: np.ndarray      # SABR vols on the grid
    F: float
    T: float
    r: float

    # ---- summary statistics -------------------------------------------------
    def prob_above(self, level: float) -> float:
        """P(S_T > level)."""
        return float(1.0 - np.interp(level, self.strikes, self.cdf))

    def prob_below(self, level: float) -> float:
        return float(np.interp(level, self.strikes, self.cdf))

    def prob_between(self, lo: float, hi: float) -> float:
        return float(np.interp(hi, self.strikes, self.cdf) - np.interp(lo, self.strikes, self.cdf))

    def quantile(self, p: float) -> float:
        """Inverse CDF."""
        return float(np.interp(p, self.cdf, self.strikes))

    def mean(self) -> float:
        return float(_trapz(self.strikes * self.pdf, self.strikes))

    def mode(self) -> float:
        return float(self.strikes[int(np.argmax(self.pdf))])

    def std(self) -> float:
        m = self.mean()
        var = _trapz((self.strikes - m) ** 2 * self.pdf, self.strikes)
        return float(np.sqrt(max(var, 0.0)))

    def stats(self) -> dict:
        return {
            "forward": self.F,
            "mean": self.mean(),
            "mode": self.mode(),
            "std": self.std(),
            "p05": self.quantile(0.05),
            "p25": self.quantile(0.25),
            "median": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p95": self.quantile(0.95),
        }


def extract_rnd(
    params: SABRParams,
    r: float = 0.0,
    n_grid: int = 800,
    k_lo_mult: float = 0.30,
    k_hi_mult: float = 2.50,
    strike_lo: float | None = None,
    strike_hi: float | None = None,
) -> RiskNeutralDensity:
    """
    Extract the risk-neutral density from a calibrated SABR smile.

    The PDF is obtained as the discounted second derivative of the call price
    with respect to strike, computed by central finite differences on a dense,
    uniform strike grid. Because prices come from the *fitted* SABR curve the
    second derivative is smooth and non-negative (up to tiny numerical noise,
    which we clip and renormalise).

    Parameters
    ----------
    params : calibrated SABRParams for the target expiry
    r : continuously-compounded discount rate over T (for e^{rT} factor)
    n_grid : number of strike grid points
    k_lo_mult, k_hi_mult : grid bounds as multiples of the forward (used only if
        explicit strike_lo/strike_hi not given)
    strike_lo, strike_hi : explicit grid bounds in underlying units
    """
    F = params.F
    T = params.T
    shift = params.shift

    lo = strike_lo if strike_lo is not None else k_lo_mult * F
    hi = strike_hi if strike_hi is not None else k_hi_mult * F
    # For rates near zero, widen sensibly.
    if hi <= lo:
        lo, hi = F - 3.0, F + 3.0

    K = np.linspace(lo, hi, n_grid)
    dK = K[1] - K[0]

    vols = params.vol(K)
    C = black76_call(F, K, T, vols, r, shift=shift)

    # Second derivative via central differences.
    d2C = np.empty_like(C)
    d2C[1:-1] = (C[2:] - 2.0 * C[1:-1] + C[:-2]) / dK**2
    d2C[0] = d2C[1]
    d2C[-1] = d2C[-2]

    pdf = np.exp(r * T) * d2C
    # Clip small negatives from numerical noise, then renormalise to integrate 1.
    pdf = np.clip(pdf, 0.0, None)
    area = _trapz(pdf, K)
    if area > 0:
        pdf = pdf / area

    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dK)])
    # Normalise cdf endpoint to 1.
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    return RiskNeutralDensity(
        strikes=K, pdf=pdf, cdf=cdf, call_prices=C, fitted_vols=vols,
        F=F, T=T, r=r,
    )
