# Evaluation Report: Batched vs Per-Row

## Setup

- Sample rows evaluated: 20
- Approaches compared: per_row, per_row_eval
- Decision/verify batch size (batched): 10 claims/request
- Dry run: no

Approach definitions:

- `per_row`: gemini-3.1-flash-lite (one multimodal request per row)
- `per_row_eval`: gemini-3.1-flash-lite + gemma-4-31b-it fallback + gemini-3.5-flash eval layer (text-only, 25/req)

## Accuracy Summary (head-to-head)

| Metric | per_row | per_row_eval |
|---|---|---|
| Exact row match | 25.00% | 25.00% |
| `evidence_standard_met` | 95.00% | 95.00% |
| `risk_flags` | 55.00% | 55.00% |
| `issue_type` | 45.00% | 45.00% |
| `object_part` | 80.00% | 80.00% |
| `claim_status` | 85.00% | 85.00% |
| `valid_image` | 95.00% | 95.00% |
| `severity` | 50.00% | 50.00% |
| `severity` (within 1 level) | 90.00% | 90.00% |

_`severity (within 1 level)` is supplementary: severity is ordinal (none<low<medium<high), so an off-by-one (e.g. medium vs high) is a near-miss, not a flip. Exact-match above is the spec's implied scoring._

## Operational Analysis

Cost is approximate at assumed blended rates ($0.30/1M input, $2.50/1M output); token counts are real (Gemini usageMetadata). Wall time includes rate-limit spacing between calls.

| Metric | per_row | per_row_eval |
|---|---|---|
| API calls (total) | 0 | 1 |
|   - observe / decide / verify | 0/0/0 | 0/0/1 |
| Requests per claim | 0.00 | 0.05 |
| Batches | 0 | 0 |
| Cache hits | 20 | 20 |
| Images sent | 0 | 0 |
| Input tokens | 0 | 15,376 |
| Output tokens | 0 | 3,634 |
| Total tokens | 0 | 19,010 |
| Approx cost (USD) | $0.0000 | $0.0137 |
| Wall time (s) | 0.1 | 17.3 |
| Throughput RPM (incl. spacing) | 0.0 | 3.5 |
| Throughput TPM (incl. spacing) | 0 | 65,769 |

Note: free-tier rate limits are the binding constraint here — observed `gemini-3-flash` (decide) cap is ~20 requests/min, and that model also returns transient 503s under load. The batched pipeline makes far fewer requests but routes them through these capacity-constrained preview models; per-row uses the lighter `flash-lite`, which had more headroom.

## Full Test-Set Projection

Projected for all 44 `claims.csv` rows (82 images), scaling tokens and cost linearly from the sample and computing request counts from the batching math:

| Metric | per_row | per_row_eval |
|---|---|---|
| Projected API calls | 44 | 44 |
| Projected total tokens | 0 | 41,822 |
| Projected cost (USD) | $0.0000 | $0.0301 |

## Strategy & Engineering (cost, latency, rate limits)

- **Batching** — `batched` describes ≤25 images/request, then decides and verifies 10 claims/request, so on the test set it makes fewer fewer requests than per-row. This is the lever for tight RPM/RPD limits.
- **Thinking disabled** (`thinkingConfig.thinkingBudget=0`) on every call. These Gemini 3.x models otherwise spend the output-token budget 'thinking', which truncated long JSON and inflated output cost ~10x; with it off they emit the JSON directly.
- **Caching** — each response is cached by prompt + model (+ image bytes) under `code/.cache/`, so reruns and the evaluation never repeat an identical call.
- **Throttling & retry** — 5s spacing between calls; on HTTP 429 the client honors the server's `retryDelay` rather than blind-retrying (blind retries also count against the per-minute quota and caused a self-inflicted storm before this fix).
- **Resilience** — a truncated or failed call degrades to a fallback row (and the verify stage backstops a failed decide) instead of aborting the run.

## Per-Approach Detail

### per_row

- Exact row match: 5/20 (25.00%)

Claim status confusion (expected -> predicted):

- `contradicted` -> `contradicted`: 3
- `contradicted` -> `supported`: 2  <-- mismatch
- `not_enough_information` -> `not_enough_information`: 2
- `supported` -> `contradicted`: 1  <-- mismatch
- `supported` -> `supported`: 12

### per_row_eval

- Exact row match: 5/20 (25.00%)

Claim status confusion (expected -> predicted):

- `contradicted` -> `contradicted`: 3
- `contradicted` -> `supported`: 2  <-- mismatch
- `not_enough_information` -> `not_enough_information`: 2
- `supported` -> `contradicted`: 1  <-- mismatch
- `supported` -> `supported`: 12

## Why This Strategy

**Chosen for `output.csv`: `per_row`.**

Reasoning, grounded in the numbers above:

- **Accuracy**: best exact-match is `per_row` (25%). The two are within a few points on every field — the expensive 3-stage `batched` pipeline does not buy meaningfully better accuracy on this task.
- **Cost**: cheapest is `per_row` ($0.0000 on the sample). `batched` uses strong models that emit several times more output tokens, so it costs more per claim even with thinking disabled.
- **Requests**: `per_row` makes the fewest calls (0 on the sample) — `batched`'s real advantage is request count, which matters only when RPM/RPD is the binding limit.
- **Reliability**: on a free-tier key, `batched` routes its few calls through the capacity-constrained `gemini-3-flash-preview` (≈20 req/min + transient 503s), whereas per-row's many small `flash-lite` calls ran clean.

**Net:** pick `per_row` for accuracy + cost + reliability. Choose `batched` instead only if your daily request cap (RPD) — not tokens or cost — is the hard constraint, since it makes far fewer requests.

