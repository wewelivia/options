# Option-Implied Probability Dashboard

Extract the **risk-neutral (option-implied) probability distribution** of an underlying
from the listed options market, and answer event-probability questions such as:

> *"What is the probability the Fed funds rate is above 5% by December?"*
> *"What is the probability SPX is above 6000 by year end?"*

**Method:** fit a **SABR** smile per expiry → apply **Breeden–Litzenberger** to the
*fitted* (smooth) call-price curve to recover the density → integrate to get the CDF
and the event probability.

```
Bloomberg (xbbg / blpapi)  →  Option chain  →  SABR calibration  →  Breeden–Litzenberger RND  →  P(event)
        │ (auto-fallback)                                                                              │
        └── Synthetic surfaces if no Terminal                                        FastAPI + web UI ─┘
```

---

## Quick start

```bash
cd rnd_dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd backend
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
```

Without a Bloomberg Terminal the app serves **realistic synthetic surfaces** (clearly
badged in the UI) so it is fully explorable anywhere. On the Terminal machine it uses
live data automatically — no code changes.

---

## How it works

### 1. Data layer — `backend/app/data/`
- **`bloomberg.py`** — `BloombergProvider` uses **xbbg** for the standard pulls
  (`BDP` spot/reference, `BDS` chain members, `BDH` historical vols) and is written to
  drop to **blpapi** directly for request types xbbg does not surface cleanly. A
  `MockProvider` generates arbitrage-consistent synthetic smiles per asset class.
- **`chain_builder.py`** — assembles a provider-agnostic `OptionChain` (grouping by
  expiry, computing forwards, cleaning IVs). Kept separate so the parsing logic is
  unit-testable without a Terminal.
- `get_provider()` auto-selects **live Bloomberg when reachable**, else the mock.

### 2. SABR calibration — `backend/app/core/sabr.py`
- Hagan (2002) lognormal implied-vol approximation with the Obløj (2008) refinement.
- **Shifted SABR** for low/negative-rate underlyings (Fed funds, SOFR).
- Per-expiry weighted least-squares fit of (α, ρ, ν) with β fixed by asset-class
  convention (rates/FX/cmdty β≈0.5, equities β≈1.0), multi-start to avoid local minima.
  α bounds scale with the forward level so high-priced underlyings (e.g. gold) calibrate.

### 3. Risk-neutral density — `backend/app/core/breeden_litzenberger.py`
- Prices the *fitted* SABR smile with Black-76, then
  `q(K) = e^{rT} · ∂²C/∂K²` on a dense strike grid.
- Because prices come from the fitted curve the second derivative is smooth and
  non-negative; the density is clipped for numerical noise and renormalised to integrate 1.
- Exposes `prob_above / prob_below / prob_between`, quantiles, mean, mode, std.

### 4. Event parser — `backend/app/core/event_parser.py`
- Rule-based (no LLM, fully auditable) parsing of conditions like
  *"above 6000 by December"*, *"below 4% by 2026-12-18"*, *"between 200 and 240 by Jan 2027"*.
- Handles direction (above/below/between), thresholds (with `%` detection for rates),
  and target dates (month names, "year end", ISO dates, "in N months").

### 5. API — `backend/app/main.py`
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness + active data source (bloomberg/mock) |
| GET | `/api/presets` | Known demo underlyings + example conditions |
| GET | `/api/chain?underlying=…` | Expiries, forwards, strike counts |
| POST | `/api/distribution` | **Main call** — full RND + event probability |

`POST /api/distribution` body:
```json
{ "underlying": "SPX Index", "condition": "above 6000 by December",
  "beta": null, "r": 0.0, "force_percent": null, "expiry": null }
```

### 6. Frontend — `frontend/`
Single-page dashboard (vanilla JS + Plotly): underlying + condition inputs, advanced
overrides (expiry, β, discount rate, %-threshold), a large probability readout with
implied odds, and four charts — risk-neutral **PDF** (with the event region shaded),
**CDF**, and **market-vs-SABR smile** — plus distribution & fit statistics.

---

## Going live on the Terminal

On the Bloomberg machine, install `xbbg` and `blpapi`, start the Terminal (DAPI on
`localhost:8194`), then run uvicorn as above. `get_provider()` detects the connection
and switches to live chains automatically. To force the mock while on the Terminal,
call `/api/chain?...&prefer_live=false`.

**Underlying identifiers** are passed straight to Bloomberg, e.g. `SPX Index`,
`AAPL US Equity`, `EURUSD Curncy`, `XAU Curncy`, or an options root for chain expansion.
For rates, point at the instrument whose options you want (Fed funds / SOFR futures
options) and express the condition in the same units (percent).

---

## Validation

`notebooks/sabr_validation.ipynb` is the QA workbench (not the deliverable). It confirms:

- **Flat-smile benchmark**: RND-implied `P(S>K)` matches the analytic Black `N(d₂)` to
  ~1e-5; PDF integrates to 1.000; `E[S_T] = F` (martingale).
- **Arbitrage checks**: density ≥ 0, CDF monotone in [0,1], call prices decreasing and
  convex in strike (no butterfly/call-spread arbitrage).
- Per-asset-class smile fits with fit RMSE and RND shape.

Run tests:
```bash
python tests/test_core_math.py     # analytic benchmarks
python tests/test_pipeline.py      # end-to-end across all asset classes
```

---

## Caveats (for the strategist)

- These are **risk-neutral** probabilities: they embed risk premia and are **not**
  physical/real-world forecasts. The gap (variance risk premium, etc.) matters for rates
  and equity-index tails in particular.
- Single-expiry RNDs answer *"at expiry T"* questions. "By December" is treated as the
  nearest listed expiry to the target date — for true *first-passage* ("touches X at any
  point before T") probabilities you need a barrier/American treatment, not the terminal law.
- SABR β is a modelling choice (fixed by convention here). Override it in the UI to test
  sensitivity.
