# Performance

Every number below was measured, not estimated — see "How these numbers
were produced" at the bottom for the exact commands, so anyone can
reproduce or challenge them. Where a run surfaced a real bug, that bug
and its fix are reported here too, not quietly folded away.

## Test hardware

Whatever machine happened to be running this project during development
— not a dedicated benchmark box, and not representative of a real
production instance. Treat every number as a floor for "does this work
at all," not a ceiling for "this is peak capacity."

| | |
|---|---|
| CPU | Intel Core i5-10210U (4 cores / 8 threads, 1.60 GHz base) |
| RAM | ~7.8 GB |
| OS | Windows 11 Home Single Language, 64-bit |
| Python | 3.13.14 |
| Redis | not configured for this run (in-process fallback stores) |
| Model | `models/risk_model.joblib`, 98 features, the one this README's own numbers are computed from |

## A bug this load test found

The very first load test run — 20 concurrent users against
`POST /api/score-custom` — came back with **every single successful
request (30/30) returning a 500**, underneath a wall of 429s from the
rate limiter. Not a load-related edge case: a single manual request
reproduced it 1-for-1.

Root cause: `RiskExplainer.score_transaction()` builds a fresh row via
`{c: txn.get(c, None) for c in feature_cols}` for any feature the caller
didn't supply. A single-row `pandas.DataFrame` column that is entirely
`None` infers as `object` dtype, not `float64`/NaN — and XGBoost's
predict path rejects `object` outright, even though the value is just a
missing marker. `POST /api/score` never hit this: its transactions come
from an already-typed multi-row DataFrame (`sample_df.iloc[idx]
.to_dict()`), where missing numeric fields are already real NaN.
`POST /api/score-custom` builds its row from scratch every time — and
per `CustomTransactionRequest`, only `TransactionAmt` is required, so
in practice *every* real call left dozens of numeric features unset.
The entire test suite mocks `RiskExplainer` for API tests, so nothing
had ever scored a hand-built transaction against the real model before
this load test did.

**Fixed** in `src/risk_explainer.py` (and the identical pattern in
`src/shadow_scoring.py`'s `ShadowScorer`) by coercing every non-
categorical column to numeric after the categorical re-cast:
`X[numeric_cols].apply(pd.to_numeric, errors="coerce")` — `None` becomes
`NaN`, which is exactly the missing-value representation XGBoost's own
native handling already expects. A regression test
(`test_missing_numeric_field_regression` in `tests/test_risk_explainer.py`)
reproduces it against a real (tiny, synthetic) trained model, confirmed
to fail without the fix and pass with it.

All numbers below were measured **after** this fix.

## Request latency: `POST /api/score-custom`

25 sequential real requests against the real model (`TransactionAmt`
plus a couple of optional fields, most features left unset — the
realistic shape of a hand-entered or lightly-populated transaction),
staying inside the 30-requests/minute-per-key budget so none of them hit
the rate limiter:

| | |
|---|---|
| Success rate | 25/25 (100%) |
| min | 172 ms |
| p50 | 184 ms |
| p90 | 196 ms |
| p99 / max | 293 ms |

This is the full request: SHAP explanation (`top_factors`), entity
escalation lookup, review-queue insertion when flagged, the escalation-
alert check, domain-metric counters, and the audit-log append — every
piece RT-1 through RT-9 added, running for real, not mocked.

## Where the time actually goes

In-process benchmarks (no HTTP, no rate limiter — isolating the
computation itself), 60 iterations each, single-threaded on one core:

| Path | p50 | Throughput |
|---|---|---|
| `RiskExplainer.score_transaction()` alone | 164 ms | 6.0 scores/sec |
| Full `ScoringService.score_and_decide()` (entity memory, review queue, escalation-alert check, audit log) | 172 ms | 5.8 scores/sec |

**SHAP is the bottleneck, not this project's own additions.** Everything
built across RT-1–RT-9 — entity/escalation tracking, the review queue,
escalation-transition alerting, and the hash-chained audit log — adds
roughly **8 ms (~5%)** on top of the raw model + SHAP call. That's not a
coincidence: it's exactly why shadow scoring (`src/shadow_scoring.py`)
deliberately skips SHAP entirely for the candidate model — SHAP's
`TreeExplainer` construction plus a `shap_values()` call per transaction
is the expensive part of scoring by a wide margin, and this benchmark is
the number that claim was resting on.

## `GET /api/health`

50 sequential requests, unauthenticated (by design — see
`api/main.py`'s docstring):

| | |
|---|---|
| min | 2.3 ms |
| p50 | 2.6 ms |
| p90 | 3.0 ms |
| p99 / max | 216 ms (one outlier; every other sample was under 5 ms) |

Cheap enough to be a real load-balancer liveness probe, which is the
point of it staying unauthenticated and off the scoring path.

## The rate limit is a policy, not a capacity ceiling

`POST /api/score` and `POST /api/score-custom` are limited to
**30 requests/minute per API key** (`api/main.py`, keyed by `X-API-Key`
— see "Rate limiting" above in the README). The very first load test run
demonstrated this working exactly as designed: with 20 concurrent
simulated users sharing one key, 3181 of 3211 requests in 60 seconds
came back 429, and every one of the 30 that got through in that window
landed inside the budget. That's deliberate — this endpoint scores
against a real model, evaluates entity escalation, and eventually earns
a downstream Gemini call — and 30/min is a per-*key* budget, not a
whole-service one: a deployment serving many distinct callers scales
roughly linearly with active keys, up to whatever the underlying
compute (measured above) can actually sustain.

No test here bypasses that limiter to find its raw ceiling — doing so
would mean running the production code path in a state it's never
actually deployed in, which would produce a number this project has no
way to honestly claim represents anything real. The in-process
benchmarks above are the closest honest substitute: they show what the
computation costs per call, decoupled from the policy choice of how
often any one caller is allowed to make it.

## How these numbers were produced

```bash
# Server (SKIP_FEATURE_SEED keeps startup fast; no Redis, no Gemini key
# needed — explanations degrade to a fast fallback without one):
API_KEY=<key> SKIP_FEATURE_SEED=1 uvicorn api.main:app --host 127.0.0.1 --port 8010

# Load test tool (ops/load/locustfile.py) — what surfaced the bug above:
pip install -r requirements-dev.txt
API_KEY=<key> locust -f ops/load/locustfile.py --host http://127.0.0.1:8010 \
    --headless --users 20 --spawn-rate 5 --run-time 60s
```

The specific latency percentiles quoted above came from short, direct
Python scripts (`urllib`/in-process timing loops, no extra dependency)
rather than locust's own output — after the rate limiter turned a
60-second locust run into mostly 429s, a clean sequential measurement
inside the actual budget was the only way to get real per-request
latency numbers instead of a distribution dominated by requests that
never reached the model at all. `ops/load/locustfile.py` remains the
right tool for reproducing the rate-limit behavior itself, or for
pointing at a deployment with multiple real API keys to exercise
aggregate throughput honestly.
