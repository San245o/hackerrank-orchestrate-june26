# Test-Set Run Report — `output.csv`

Final predictions for all rows of `dataset/claims.csv` using the full pipeline:
primary **gemini-3.1-flash-lite** (per-row) → **gemma-4-31b-it** fallback (if primary fails,
separate quota) → **gemini-3.5-flash** text-only eval-layer critic (batched 25 rows/request).

## Run configuration

| Item | Value |
|---|---|
| Pipeline | `per_row` (direct) with fallback + eval layer |
| Primary model | `gemini-3.1-flash-lite` |
| Fallback model | `gemma-4-31b-it` (triggered only on primary failure) |
| Eval layer | `gemini-3.5-flash`, text-only, 25 rows/request |
| Prompt set | `orchestrate-gemini-v7` |
| Thinking | disabled on all models (`thinkingConfig.thinkingBudget=0`; Gemma omits it) |
| Request spacing | 5 s |
| Rows processed | 44 / 44 |
| Output | `../dataset/output.csv` (quote-all, matches provided template) |
| Command | `python3 code/main.py --pipeline direct --strategy flash_lite --fallback-strategy gemma_4 --eval-strategy flash_35 --eval-batch-size 25 --input dataset/claims.csv --output dataset/output.csv` |

## Operational statistics (real, from Gemini `usageMetadata`)

Primary (flash_lite) was fully cached from the prior run. Only the 2 eval-layer calls were new.

| Metric | Primary (cached) | Eval layer (new) | Total |
|---|---|---|---|
| API calls | 0 (44 cache hits) | 2 | 2 |
| Fallback calls | 0 | — | 0 |
| Failed calls | 0 | 0 | 0 |
| Images sent | 0 (cached) | 0 (text-only) | 82 (original run) |
| Input tokens | 0 | 33,442 | ~218,437 |
| Output tokens | 0 | 8,323 | ~15,512 |
| Total tokens | 0 | 41,765 | ~233,949 |
| Approx cost | $0.0735 (cached) | $0.0307 | ~$0.1042 |
| Wall time | — | 40.2 s | ~396 s (incl. original) |

*Primary row counts from the original run: 184,995 input + 7,189 output = 192,184 tokens, $0.0735.*
Pricing assumed: $0.30/1M input, $2.50/1M output. Token counts are real (usageMetadata).

## Output summary (`output.csv`, 44 rows)

- Schema: 14 columns, exact order per `problem_statement.md`, every field quoted.
- Every row has a non-empty `claim_status_justification`. No parse/error placeholders.
- `claim_status`: 17 `supported`, 26 `contradicted`, 1 `not_enough_information`.
- `severity`: none 25, medium 8, high 5, low 4, unknown 2.
- `damage_not_visible` in risk_flags: **14 rows** (eval layer added this on contradiction rows
  where the part was visible but the claimed damage was absent — up from 6 without the eval layer).
- `manual_review_required`: 30 rows (from explicit `history_flags=user_history_risk`).

## What the eval layer changed (test set)

The eval layer ran 2 batches (25 + 19 rows) on top of the primary predictions:
- Corrected `damage_not_visible` flagging on contradiction rows (rubric enforcement).
- Made no changes to `claim_status` (the decision field).
- On the 20-row labeled sample: 18 agree, 2 corrected; accuracy unchanged at 25% exact-match
  (the 2 corrections hit gold-label ambiguities: `damage_not_visible` vs `claim_mismatch`,
  `none` vs `unknown` — either answer is defensible at n=20).

## Rate-limit / reliability notes

- All calls succeeded. `flash-lite` used its own cache; eval layer's `flash_35` ran cleanly.
- `gemma-4-31b-it` fallback was not triggered (primary never failed).
- Responses cached under `code/.cache/` so re-running is free and idempotent.

## Gemini logs

Full operational trace: `code/logs/test_run_gemini_log.txt`.
Raw per-row responses cached under `code/.cache/flash_lite/<hash>.json`.
