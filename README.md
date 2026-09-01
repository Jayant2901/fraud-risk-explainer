# AI Risk Manager — Razorpay Buildathon (Track 2)

An entity-aware transaction risk system: an XGBoost model scores each
transaction, SHAP explains which signals drove the score, and a
**Gemini-powered LLM agent** (via the Google Gen AI API, free tier)
reasons over both the transaction and the entity's (card/account)
recent risk trajectory to produce a human-readable explanation — with
the threshold itself chosen to minimize business cost, not just
maximize accuracy.

## Why this is different from a standard fraud classifier

Most fraud-detection submissions stop at: train a classifier, wrap it
in SHAP, done. This project makes several deliberate additions, each
solving a real gap in that standard approach:

1. **Entity risk memory (the "agentic" piece).** Transactions aren't
   scored in isolation. Each card/account fingerprint (`entity_id`) has
   a rolling verdict history; three or more REVIEW/BLOCK verdicts in
   its recent window pushes it into a `WATCH` or `ELEVATED` escalation
   state. The LLM agent is given this state explicitly and can escalate
   its recommended action beyond what the raw score alone suggests —
   and must say so, rather than silently overriding the model. See
   `src/entity_memory.py`.

2. **Cost-optimal threshold, not just AUC.** A risk team doesn't
   optimize accuracy — they minimize expected cost, where missed fraud
   and wrongly-blocked legitimate customers have very different costs.
   `src/cost_analysis.py` sweeps thresholds and picks the one that
   minimizes `false_negatives * avg_fraud_loss + false_positives *
   avg_fp_cost`, and reports the estimated savings vs. a naive 0.5
   threshold. The cost assumptions are explicit, adjustable constants —
   not hidden inside a single "accuracy" number.

3. **Structured LLM output, not scraped JSON.** `src/llm_agent.py` calls
   Gemini (`gemini-3.6-flash`) through the Google Gen AI SDK's
   structured outputs (`response_schema=RiskVerdict`, a Pydantic model,
   read back via `response.parsed`), so the action is schema-constrained
   to `ALLOW`/`REVIEW`/`BLOCK` at generation time rather than hoping the
   model's free-text JSON parses. Untrusted transaction fields (email
   domain, product code, etc.) are sanitized before being interpolated
   into the prompt, and the system prompt explicitly instructs the
   model to treat them as data, never as instructions — a basic
   prompt-injection defense, since those fields ultimately trace back
   to attacker-controlled transaction data.

4. **The LLM is never on the authorization critical path.** `POST
   /api/score` makes the ALLOW/REVIEW/BLOCK decision synchronously from
   the score and a deterministic rule table (`decide_action` in
   `api/main.py`) and returns immediately — this is what actually gates
   the transaction. Measured locally on CPU, that decision path (XGBoost
   predict + SHAP + rules) takes ~100-130ms; the Gemini API call it
   *doesn't* wait on adds real network + generation latency on top of
   that (and free-tier requests can additionally queue behind a rate
   limit). The LLM runs afterward as a FastAPI background task purely to
   generate a human-readable explanation for the reviewer queue; the
   frontend polls `GET /api/explanations/{verdict_id}` for it. A
   real-time payment authorization path can't block on an LLM API call
   (latency, and a rate limit or outage on the vendor's side
   shouldn't take down transaction authorization), so this mirrors how
   such systems are actually built — decision first, explanation
   enrichment after.

5. **Idempotency on the scoring endpoint.** `POST /api/score` accepts an
   optional `Idempotency-Key` header (the same convention Razorpay's own
   API asks integrators to use). A retried request with the same key
   returns the original cached response instead of re-scoring — without
   this, a network retry or a double-submit would score the same
   transaction twice and double-count it in the entity's escalation
   history, incorrectly pushing them toward `WATCH`/`ELEVATED`.

## A note on the dataset and the entity fingerprint

IEEE-CIS is a genuinely anonymized dataset — it deliberately does not
ship a raw account/merchant ID. `src/data_utils.py` builds a proxy
entity fingerprint from `card1 + card2 + card5 + addr1 +
P_emaildomain`. This is the well-known "UID" technique used in several
top-performing solutions to the original Kaggle competition, not
something invented for this project — it mirrors how real card
networks do entity resolution (device + card + address fingerprinting)
when no explicit account ID is available in the event stream. With
real Razorpay data, you'd use the actual merchant/account ID directly.

## Setup

```bash
pip install -r requirements.txt
```

### 1. Get the data

This is a **Kaggle competition** dataset — you must join the
competition once (free, one click) before the API will let you
download it:
https://www.kaggle.com/competitions/ieee-fraud-detection

Then, with Kaggle API credentials configured (`~/.kaggle/kaggle.json`
or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars):

```bash
python src/download_data.py
```

This places `train_transaction.csv` and `train_identity.csv` in `data/`.

### 2. Set your Gemini API key (free)

```bash
export GEMINI_API_KEY=...          # macOS/Linux
```
```powershell
$env:GEMINI_API_KEY = "..."        # Windows PowerShell, current session
setx GEMINI_API_KEY "..."          # Windows PowerShell, permanent (needs a new terminal to take effect)
```

Get a free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
— no billing required for the free tier (rate-limited). Without this
set, `/api/score` still returns the instant ALLOW/REVIEW/BLOCK decision
(see point 4 below) — only the AI explanation panel falls back to a
"credentials missing" message.

### 3. Train the model

```bash
python src/train_model.py
```

Writes `models/risk_model.joblib`, `models/feature_cols.joblib`,
`models/optimal_threshold.joblib`, and `models/eval_report.txt`
(AUC/PR-AUC plus the cost-based threshold analysis — this is what
you'll quote in the pitch).

### 4. Launch the API backend

```bash
uvicorn api.main:app --reload --port 8000
```

This serves the scoring/explanation/cost-analysis logic as a JSON API
at `http://localhost:8000/api/...`.

**First run is slow, and that's expected:** importing pandas/xgboost/shap
takes ~20-30s before the server is even listening, and the very first
`/api/entities` request loads and feature-engineers the full ~590K-row
CSV (a couple of minutes) before caching the small sample it actually
serves to `models/sample_data_cache.pkl`. Every request after that —
including future restarts, as long as that cache file exists — is fast
(well under a second). Don't open the frontend until the backend
terminal prints `Application startup complete.`, or you'll see
connection errors that look like a bug but are really just "not ready
yet."

### 5. Launch the frontend

```bash
cd frontend
npm install
npm run dev
```

Opens the app at `http://localhost:5173` (Vite dev server proxies
`/api/*` to the backend on port 8000 — see `frontend/vite.config.ts`).
For a production build: `npm run build` (outputs static assets to
`frontend/dist/`), served by any static host or reverse-proxied behind
the FastAPI backend.

**Live Scoring tab:** pick an entity, walk through its transactions in
sequence, and watch the escalation state build up as verdicts
accumulate — this is the part to demo live, since it's the piece a
static classifier genuinely can't do. The automated decision (action +
escalation flag) appears instantly; the "AI Reviewer Explanation" panel
below it fills in a moment later once the background LLM call finishes
— that gap is intentional, not a loading bug (see point 4 above).

**Cost-Optimal Threshold tab:** shows the eval report and lets you
adjust the fraud-loss/false-positive-cost assumptions to see how the
optimal threshold shifts.

### Running tests

```bash
python -m unittest discover -s tests
```

Covers `llm_agent.py`'s prompt sanitization (the prompt-injection
defense) and every failure path against a mocked Gemini client —
missing/invalid API key, rate limiting, server errors, malformed
schema — each asserting it degrades to a safe `REVIEW` fallback instead
of crashing. No API key or network access needed to run these.

## Project structure

```
risk-manager/
├── data/                        <- train_transaction.csv, train_identity.csv
├── models/                      <- generated by train_model.py
├── src/
│   ├── download_data.py         <- pulls IEEE-CIS via kagglehub
│   ├── data_utils.py            <- merge, entity fingerprint, causal features, time split
│   ├── cost_analysis.py         <- threshold -> business cost curve
│   ├── train_model.py           <- trains XGBoost, picks cost-optimal threshold
│   ├── risk_explainer.py        <- SHAP wrapper: score -> top factors
│   ├── entity_memory.py         <- rolling verdict history -> escalation state
│   └── llm_agent.py             <- Gemini (Google Gen AI API) agent, reasons over score + history
├── api/
│   └── main.py                  <- FastAPI JSON API wrapping the src/ modules
├── tests/
│   └── test_llm_agent.py        <- llm_agent.py unit tests (mocked Gemini client)
└── frontend/                    <- React + TypeScript + Tailwind SPA
    └── src/
        ├── api/client.ts        <- typed fetch client for the backend
        ├── components/
        │   ├── LiveScoring.tsx
        │   └── CostAnalysis.tsx
        └── App.tsx
```

## Methodology notes (for the writeup / judges)

- **No leakage:** entity history features (`entity_prior_fraud_rate`,
  etc.) are computed using only transactions strictly earlier in time
  than the current one — a naive full-dataset aggregate would leak the
  label. Train/test is a **chronological** split (train on the earlier
  80% of transactions, test on the later 20%), not random, for the
  same reason.
- **Categorical handling:** XGBoost's native `enable_categorical=True`
  is used for fields like `ProductCD`, `card4`, `card6`,
  `P_emaildomain` rather than manual one-hot encoding — simpler and
  handles missingness natively, which this dataset has a lot of.
- **What I'd build next** given more time: persist entity memory in
  Redis instead of in-process memory (so it survives restarts and
  scales across workers), extend the entity fingerprint with device/IP
  graph features for fraud-ring detection, and replace the fixed
  cost constants with ones pulled from real ops data.
