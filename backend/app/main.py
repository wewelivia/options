"""
FastAPI application: option-implied event-probability dashboard.

Endpoints
---------
GET  /api/health            -> liveness + data source (bloomberg | mock)
GET  /api/presets           -> known demo underlyings grouped by asset class
GET  /api/chain             -> option chain summary (expiries, forwards)
POST /api/distribution      -> full RND + event probability (the main call)
GET  /                      -> serves the single-page frontend

Run:  uvicorn app.main:app --reload --port 8000   (from backend/)
"""
from __future__ import annotations

import logging
import os
import traceback

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models.schemas import ProbabilityRequest, ChainResponse, DistributionResponse
from .core import service
from .data.bloomberg import MockProvider, get_provider

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rnd")

app = FastAPI(title="Option-Implied Probability Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
FRONTEND_DIR = os.path.abspath(FRONTEND_DIR)


@app.get("/api/health")
def health():
    prov = get_provider(prefer_live=True)
    source = "mock" if isinstance(prov, MockProvider) else "bloomberg"
    return {"status": "ok", "data_source": source,
            "note": "Live Bloomberg used automatically when xbbg/blpapi can connect; "
                    "otherwise synthetic surfaces are served so the tool is fully explorable."}


@app.get("/api/presets")
def presets():
    groups: dict[str, list[str]] = {}
    for name, (ac, _spot) in MockProvider.PRESETS.items():
        groups.setdefault(ac, []).append(name)
    examples = {
        "SPX Index": "above 6000 by December",
        "AAPL US Equity": "between 200 and 240 by Jan 2027",
        "EURUSD Curncy": "above 1.12 by year end",
        "XAU Curncy": "above 2500 by December",
        "FEDFUNDS": "above 5% by December",
    }
    return {"groups": groups, "example_conditions": examples}


@app.get("/api/diagnose")
def diagnose(underlying: str = Query(...)):
    """Step-by-step probe of the live Bloomberg path so we can see exactly which
    xbbg call fails and what it returns. Safe to call anytime; never raises --
    even a bug in the diagnostic itself is returned as JSON, not a 500."""
    import json, math

    def _scrub(o):
        """Recursively replace NaN/Inf (not JSON-compliant under starlette's
        strict encoder) with None, and stringify anything exotic."""
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict):
            return {str(k): _scrub(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_scrub(v) for v in o]
        if isinstance(o, (str, int, bool)) or o is None:
            return o
        return str(o)

    try:
        report = service.diagnose(underlying)
        return _scrub(report)
    except Exception as e:
        return {"diagnose_crashed": True,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc().splitlines()[-12:]}


@app.get("/api/chain", response_model=ChainResponse)
def chain(underlying: str = Query(...), prefer_live: bool = True):
    try:
        return service.get_chain_info(underlying, prefer_live=prefer_live)
    except Exception as e:
        # Log the full traceback to the server console so the real xbbg error
        # is visible, and return a useful message to the UI.
        log.error("Chain build failed for %r:\n%s", underlying, traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Chain error: {e}")


@app.post("/api/distribution", response_model=DistributionResponse)
def distribution(req: ProbabilityRequest):
    try:
        return service.compute_distribution(
            underlying=req.underlying,
            condition=req.condition,
            beta=req.beta,
            r=req.r,
            force_percent=req.force_percent,
            expiry=req.expiry,
        )
    except ValueError as e:
        log.warning("Distribution 422 for %r / %r: %s", req.underlying, req.condition, e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error("Distribution failed for %r / %r:\n%s",
                  req.underlying, req.condition, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")


# --- static frontend ---
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
