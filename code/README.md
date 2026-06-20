# HackerRank Orchestrate — Solution

A Gemini-only visual-evidence-review pipeline. For each row in `dataset/claims.csv` it sends
only the relevant context (claim conversation, that row's images, the matching user-history
row, and relevant evidence requirements) and writes the required `output.csv` schema.

It implements **two strategies** that are benchmarked head-to-head on the labeled
`dataset/sample_claims.csv`:

| Strategy | Pipeline | Models | Requests |
|---|---|---|---|
| **A — batched** | observe → decide → verify, all batched | describe `gemini-3.5-flash` (≤25 images/req) → decide `gemini-3-flash-preview` (10 claims/req) → verify `gemini-3.5-flash` (text-only, 10/req) | ~6 for 20 rows |
| **B — per-row** | one multimodal request per claim | `gemini-3.1-flash-lite` | 1/row |

Strategy A minimizes total **requests** (the RPM/RPD saver); Strategy B is the simple,
robust baseline.

## Setup

Free Google AI Studio / Gemini API key. Put it in `.env` at the repo root:

```bash
echo 'GEMINI_API_KEY=your_key_here' > .env
```

Pure Python standard library — no `pip install` needed. Load the key before any real run:

```bash
set -a && . ./.env && set +a
```

> **Thinking is disabled** (`thinkingConfig.thinkingBudget: 0`) on every call. These Gemini
> 3.x models are reasoning models that otherwise burn large amounts of "thinking" tokens
> (slow, ~10× the output-token cost, and they truncate long JSON). With it off they emit the
> JSON directly. Raise `Strategy.thinking_budget` if a stage needs reasoning.

## Run final predictions

Strategy A (batched, fewest requests):

```bash
python3 code/main.py --pipeline batched --output dataset/output.csv
```

Strategy B (per-row baseline):

```bash
python3 code/main.py --pipeline direct --strategy flash_lite --output dataset/output.csv
```

Wiring check without API calls:

```bash
python3 code/main.py --pipeline batched --dry-run --limit 4 --output /tmp/dry.csv
```

## Evaluate (compare both on the labeled sample)

```bash
python3 code/evaluation/main.py --approaches batched,per_row --max-api-calls 80
```

Writes `code/evaluation/sample_predictions_<approach>.csv`, `metrics_<approach>.csv`, and a
head-to-head `evaluation_report.md` with accuracy + a data-driven operational analysis
(requests, real tokens from `usageMetadata`, approx cost, wall time, throughput RPM/TPM).

## Key flags

- `--pipeline {direct,batched}` — direct = per-row (B); batched = 3-stage (A)
- `--decision-batch-size` / `--verify-batch-size` — claims per batched request (default 10)
- `--max-images-per-chunk` — images per describe request (default 25)
- `--max-api-calls` — hard live ceiling; aborts before exceeding it
- `--limit N` — first N rows only; `--dry-run` — no API calls
- `--list-strategies` — print model ids

## Reliability / rate limits

- 5 s spacing between calls; on HTTP 429 the client honors the server's `retryDelay`
  instead of blindly retrying (blind retries also count against the quota window).
- Every stage is crash-proof: a truncated/failed call degrades to a fallback row (and the
  verify stage backstops a failed decide) rather than aborting the run.
- Responses are cached under `code/.cache/` (keyed by prompt + model + image bytes), so
  reruns don't repeat identical calls.

## Findings (20-row sample)

- **Accuracy is ~tied** (exact-row-match 15–20%; most fields within a few points). The
  expensive 3-stage pipeline does **not** beat per-row `flash-lite` here.
- **batched** makes 3.3× fewer requests (6 vs 20) — but routes them through the
  capacity-constrained `gemini-3-flash-preview` (free-tier ~20 req/min, transient 503s), and
  costs ~80% more (strong models emit ~4× the output tokens even with thinking off).
- **per-row** is cheaper, faster, higher-throughput, and more reliable on a free-tier key.
- **Recommended default for `output.csv`: per-row (`flash_lite`)**, unless the daily request
  cap (RPD) is your binding constraint, in which case batched's lower request count wins.

See `code/evaluation/evaluation_report.md` for the full numbers.
