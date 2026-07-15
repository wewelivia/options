"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ProbabilityRequest(BaseModel):
    underlying: str = Field(..., description="Bloomberg ticker or preset, e.g. 'SPX Index', 'FEDFUNDS'")
    condition: str = Field(..., description="Natural-language event, e.g. 'above 6000 by December'")
    # Optional overrides
    beta: Optional[float] = Field(None, description="SABR beta override (else asset-class default)")
    r: float = Field(0.0, description="Discount rate for the e^{rT} factor")
    force_percent: Optional[bool] = Field(None, description="Treat threshold as percent (rates)")
    expiry: Optional[str] = Field(None, description="Force a specific expiry ISO date; else nearest to target")


class ExpiryInfo(BaseModel):
    expiry: str
    T: float
    forward: float
    n_strikes: int


class ChainResponse(BaseModel):
    underlying: str
    asset_class: str
    spot: float
    as_of: str
    source: str
    shift: float
    expiries: list[ExpiryInfo]


class SmilePoint(BaseModel):
    strike: float
    market_vol: float
    fitted_vol: float


class DistributionResponse(BaseModel):
    underlying: str
    asset_class: str
    source: str
    expiry: str
    T: float
    forward: float
    is_percent: bool
    # SABR fit
    sabr: dict
    smile: list[SmilePoint]
    # RND arrays (downsampled for transport)
    grid: list[float]
    pdf: list[float]
    cdf: list[float]
    stats: dict
    # Event result
    probability: float
    condition: str
    direction: str
    threshold: float
    threshold_hi: Optional[float]
    target_date: str
    # Extra readouts
    complement: float
    odds: str
