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

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models.schemas import ProbabilityRequest, ChainResponse, DistributionResponse
from .core import service
from .data.bloomberg import MockProvider, get_provider

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


@app.get("/api/chain", response_model=ChainResponse)
def chain(underlying: str = Query(...), prefer_live: bool = True):
    try:
        return service.get_chain_info(underlying, prefer_live=prefer_live)
    except Exception as e:
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
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")


# --- static frontend ---
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
