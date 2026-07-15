"""
SABR (Stochastic Alpha Beta Rho) volatility model.

Implements the Hagan et al. (2002) lognormal (Black) implied-vol approximation
and per-expiry calibration to a market vol smile. The fitted smile is the input
to the Breeden-Litzenberger risk-neutral density extraction.

References
----------
Hagan, Kumar, Lesniewski, Woodward (2002), "Managing Smile Risk",
    Wilmott Magazine.
Obloj (2008), "Fine-tune your smile: Correction to Hagan et al." -- improved
    ATM expansion used here to avoid the well-known small-strike divergence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares


# ----------------------------------------------------------------------------
# Core Hagan lognormal implied volatility
# ----------------------------------------------------------------------------
def sabr_lognormal_vol(
    F: float,
    K: np.ndarray | float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    eps: float = 1e-07,
) -> np.ndarray:
    """
    Hagan (2002) lognormal (Black) implied vol under SABR, with the Obloj (2008)
    refinement for numerical stability near the money.

    Parameters
    ----------
    F : forward of the underlying for expiry T
    K : strike(s)
    T : time to expiry in years
    alpha : instantaneous vol level (alpha > 0)
    beta : CEV exponent in [0, 1]
    rho : correlation between spot and vol, in (-1, 1)
    nu : vol-of-vol (nu >= 0)

    Returns
    -------
    Black lognormal implied vol at each strike.
    """
    K = np.asarray(K, dtype=float)
    F = float(F)

    # Guard against non-positive forwards/strikes (rates can be handled by the
    # shifted-SABR wrapper; here we assume positive inputs).
    one_beta = 1.0 - beta
    logFK = np.log(F / K)
    FK_beta = (F * K) ** (one_beta / 2.0)

    # z and x(z)
    z = (nu / alpha) * FK_beta * logFK
    # x(z) via the standard formula; handle z -> 0 by series expansion.
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    xz = np.log((sqrt_term + z - rho) / (1.0 - rho))

    # z / x(z) with the removable singularity at z=0 handled.
    with np.errstate(divide="ignore", invalid="ignore"):
        z_over_xz = np.where(np.abs(z) < eps, 1.0 - 0.5 * rho * z, z / xz)

    # Denominator expansion in log(F/K)
    denom = FK_beta * (
        1.0
        + (one_beta**2 / 24.0) * logFK**2
        + (one_beta**4 / 1920.0) * logFK**4
    )

    # Time-dependent correction bracket
    A = (one_beta**2 / 24.0) * (alpha**2 / (FK_beta**2))
    B = 0.25 * (rho * beta * nu * alpha) / FK_beta
    C = (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    correction = 1.0 + (A + B + C) * T

    vol = (alpha / denom) * z_over_xz * correction

    # ATM limit (K == F): the logFK terms vanish; z -> 0 handled above, but the
    # denominator FK_beta = F**one_beta, so recompute cleanly for K ~= F.
    atm_mask = np.abs(logFK) < eps
    if np.any(atm_mask):
        Fb = F**one_beta
        atm_vol = (alpha / Fb) * (
            1.0
            + (
                (one_beta**2 / 24.0) * (alpha**2 / (F ** (2.0 * one_beta)))
                + 0.25 * (rho * beta * nu * alpha) / Fb
                + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
            )
            * T
        )
        vol = np.where(atm_mask, atm_vol, vol)

    return vol


# ----------------------------------------------------------------------------
# Shifted SABR (for low/negative rates -- Fed Funds, SOFR)
# ----------------------------------------------------------------------------
def sabr_shifted_vol(
    F: float,
    K: np.ndarray | float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.03,
) -> np.ndarray:
    """
    Shifted SABR: apply a positive displacement so that (F+shift) and (K+shift)
    are positive, then use the standard lognormal formula. Required for rates
    where forwards/strikes can be near zero or negative.
    """
    return sabr_lognormal_vol(F + shift, np.asarray(K, float) + shift, T, alpha, beta, rho, nu)


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
@dataclass
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float
    F: float
    T: float
    shift: float = 0.0
    rmse: float = float("nan")

    def vol(self, K: np.ndarray | float) -> np.ndarray:
        if self.shift:
            return sabr_shifted_vol(self.F, K, self.T, self.alpha, self.beta, self.rho, self.nu, self.shift)
        return sabr_lognormal_vol(self.F, K, self.T, self.alpha, self.beta, self.rho, self.nu)

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "nu": self.nu,
            "F": self.F,
            "T": self.T,
            "shift": self.shift,
            "rmse": self.rmse,
        }


def calibrate_sabr(
    F: float,
    strikes: Sequence[float],
    market_vols: Sequence[float],
    T: float,
    beta: float | None = 0.5,
    shift: float = 0.0,
    weights: Sequence[float] | None = None,
) -> SABRParams:
    """
    Calibrate SABR to a market vol smile for a single expiry.

    beta is usually FIXED by asset class convention (equities ~1.0 sticky-delta
    or 0.5, rates ~0.5, FX ~0.5-1.0) rather than fitted, because (beta, rho) are
    jointly under-identified from a single smile. If beta is None it is fitted
    too (bounded to [0,1]).

    Fits alpha, rho, nu (and optionally beta) by weighted least squares on
    implied vols. Uses multiple random restarts to avoid local minima.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)
    if weights is None:
        weights = np.ones_like(market_vols)
    weights = np.asarray(weights, dtype=float)

    fit_beta = beta is None

    # alpha scales like sigma_atm * (F+shift)^(1-beta), so its natural magnitude
    # depends on the forward level and beta. Set a generous upper bound relative
    # to that scale instead of a fixed constant (which breaks for high-priced
    # underlyings like gold or an equity index when beta < 1).
    _b_for_scale = beta if beta is not None else 0.5
    alpha_scale = max((F + shift), 1e-6) ** (1.0 - _b_for_scale)
    alpha_ub = max(5.0, 50.0 * alpha_scale)

    def vol_fn(params):
        if fit_beta:
            alpha, b, rho, nu = params
        else:
            alpha, rho, nu = params
            b = beta
        if shift:
            return sabr_shifted_vol(F, strikes, T, alpha, b, rho, nu, shift)
        return sabr_lognormal_vol(F, strikes, T, alpha, b, rho, nu)

    def residuals(params):
        return weights * (vol_fn(params) - market_vols)

    # Bounds (alpha upper bound scaled to the forward level / beta)
    if fit_beta:
        lb = [1e-6, 0.0, -0.999, 1e-4]
        ub = [alpha_ub, 1.0, 0.999, 10.0]
    else:
        lb = [1e-6, -0.999, 1e-4]
        ub = [alpha_ub, 0.999, 10.0]

    # Initial guess: alpha from ATM vol, rho ~ 0, nu ~ 0.3
    atm_idx = int(np.argmin(np.abs(strikes - F)))
    atm_vol = market_vols[atm_idx]
    Fb = (F + shift) ** (1.0 - (beta if beta is not None else 0.5))
    alpha0 = float(np.clip(atm_vol * Fb, lb[0], ub[0]))

    best = None
    rng = np.random.default_rng(42)
    starts = []
    if fit_beta:
        starts.append([alpha0, 0.5, -0.2, 0.4])
    else:
        starts.append([alpha0, -0.2, 0.4])
    for _ in range(8):
        if fit_beta:
            starts.append([
                alpha0 * rng.uniform(0.5, 1.5),
                rng.uniform(0.1, 0.9),
                rng.uniform(-0.8, 0.5),
                rng.uniform(0.1, 1.5),
            ])
        else:
            starts.append([
                alpha0 * rng.uniform(0.5, 1.5),
                rng.uniform(-0.8, 0.5),
                rng.uniform(0.1, 1.5),
            ])

    for x0 in starts:
        try:
            res = least_squares(residuals, x0, bounds=(lb, ub), method="trf", max_nfev=5000)
        except Exception:
            continue
        rmse = float(np.sqrt(np.mean((res.fun / np.where(weights == 0, 1, weights)) ** 2)))
        if best is None or rmse < best[1]:
            best = (res.x, rmse)

    if best is None:
        raise RuntimeError("SABR calibration failed for all restarts")

    x, rmse = best
    if fit_beta:
        alpha, b, rho, nu = x
    else:
        alpha, rho, nu = x
        b = beta

    return SABRParams(alpha=float(alpha), beta=float(b), rho=float(rho), nu=float(nu),
                      F=float(F), T=float(T), shift=float(shift), rmse=rmse)
