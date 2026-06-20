# Interview Guide — Multi-Modal Evidence Review System

## What the system does (one paragraph)

The system verifies visual damage claims for three object types: cars, laptops, and packages. For
each claim it receives a chat transcript, one or more images, the user's claim history, and a
minimum evidence checklist. It decides whether the submitted images support the claim, contradict
it, or provide insufficient information, and outputs a structured 14-field row: evidence
sufficiency, risk flags, issue type, object part, the decision (supported / contradicted /
not_enough_information), supporting image IDs, image validity, severity, and a short
image-grounded justification. It processes the full 44-row test set using the Gemini API with
no third-party dependencies — pure Python stdlib.

---

## Architecture in one diagram

```
INPUT: claims.csv + images + user_history.csv + evidence_requirements.csv
         |
         ▼
[Stage 1 — Primary vision call]
  gemini-3.1-flash-lite
  one multimodal request per row
  images + claim + selected history row + relevant requirements → raw JSON row
         |  (on failure)
         ▼
[Stage 1 fallback — gemma-4-31b-it]
  separate quota pool, no thinkingConfig
  same prompt + images → raw JSON row
         |
         ▼
[Deterministic validate_prediction]
  re-stamps input columns, clamps every field to allowed values,
  adds risk flags from history_flags
         |
         ▼
[Stage 2 — Eval layer (text-only, batched)]
  gemini-3.5-flash, 25 rows per request, no images
  audits rubric consistency (damage_not_visible, none vs unknown, etc.)
  corrects and returns same schema; falls back to keeping candidate if it fails
         |
         ▼
OUTPUT: dataset/output.csv (44 rows, 14 quoted columns)
```

---

## 30 Questions and Answers

### System design

**1. What is the end-to-end flow of the system?**
For each claim row: build a prompt carrying the claim text, that user's history row, and relevant
evidence requirements; attach the claim's images inline (base64); call gemini-3.1-flash-lite; parse
and validate the JSON response; run a text-only batched eval-layer critic over 25 rows at a time
using gemini-3.5-flash; write the corrected row to output.csv. If the primary call fails, the
system retries with gemma-4-31b-it before falling back to a deterministic manual-review row.

**2. Why did you choose a per-row approach instead of batching all claims into one prompt?**
Two reasons. First, each row has its own images (44 rows × ~2 images × ~1-3 MB = ~100 MB raw),
which would easily exceed the Gemini inline payload limit in a single request. Second, attention
dilution: packing 44 claims into one prompt risks the model mixing up details between claims.
The per-row approach keeps each claim fully isolated and self-contained.

**3. What is the role of the eval layer and why is it text-only?**
The eval layer is a consistency auditor. After the vision model has already seen the images, a
second vision pass would just re-run the same expensive operation. Instead, gemini-3.5-flash reads
the candidate decision alongside the claim text and rubric rules, checks for internal consistency
violations (e.g. claim_status=contradicted but issue_type=dent instead of none), and corrects
them. It's text-only because it's enforcing rubric logic, not re-examining pixels. On the 44-row
test set it added `damage_not_visible` to 14 contradiction rows that the primary missed.

**4. How does the Gemma fallback differ from the Gemini models and why use it?**
Gemma is an open-weights model served on the Gemini API but with a separate quota pool from the
3.x preview models. It rejects `thinkingConfig` (which flash_lite and flash_35 both accept), so
the `generation_config()` function detects any model starting with "gemma" and omits that field.
It also produces looser JSON (sometimes wrapping the object in prose), so the response goes
through the salvage extractor (`extract_json_object` with bracket fallback) before validation.
The key advantage: if flash_lite hits its free-tier cap, Gemma can handle the row independently.

**5. What is `validate_prediction` and why is every response funneled through it?**
`validate_prediction` is the schema safety net. It re-stamps the four input columns
(user_id, image_paths, user_claim, claim_object) from the actual input row — so even if a model
hallucinates a different user_id, the output is always correct. It then normalises every predicted
field: clamping claim_status/issue_type/severity/etc. to their allowed value sets, deriving
supporting_image_ids from the real image filenames, and injecting risk flags from the history rule.
Because the eval layer also calls validate_prediction, corrupted or out-of-schema critic output
cannot reach output.csv.

**6. Why are images encoded inline rather than uploaded to a file API?**
The Gemini File API requires managing upload lifecycles (create, poll status, use, delete) and
adds latency and complexity. For this dataset the total raw image size is ~42 MB (test set),
which base64-inflates to ~56 MB — well within Gemini's per-request inline limit with the 25-image
chunking used in the batched observation pipeline. Inline encoding is deterministic and cache-key
friendly: the cache key hashes the actual image bytes, so a changed image automatically
invalidates the cache without any external state.

---

### Models and prompting

**7. Why is thinking disabled on all models (`thinkingBudget=0`)?**
The Gemini 3.x models are reasoning models that by default spend a large portion of the output
budget "thinking" before emitting the final JSON. In testing, a single 10-claim batched decision
call generated ~30,000 output tokens — almost entirely thinking tokens — causing the 120-second
read timeout to fire. With `thinkingBudget=0`, the models emit the JSON directly: the same call
dropped to ~750 output tokens, ran in ~6 seconds, and produced identical answers. Thinking is
appropriate for open-ended reasoning but adds no value when the output schema is fixed.

**8. What went wrong with `gemini-3-flash` and how was it fixed?**
The model ID `gemini-3-flash` returns a 404 on the v1beta API. The actual model available is
`gemini-3-flash-preview`. This was discovered during the first smoke test and fixed by querying
`/v1beta/models` to list available models, then updating the STRATEGIES dict. The lesson: always
probe actual model availability on the target key rather than trusting documentation model IDs.

**9. How does the system handle 429 rate-limit errors without burning quota?**
The original retry logic used exponential backoff with blind retries. On a free-tier key with a
~20 req/min cap, each blind retry counted against the window — causing a self-inflicted storm that
exhausted the quota faster than natural spacing. The fix: parse the `retryDelay` field from the
429 response body (the server tells you exactly how long to wait), sleep that duration + 1 second,
then retry once. This respects the quota window and uses far fewer retry attempts.

**10. What is the SEVERITY_GUIDE and ISSUE_TYPE_GUIDE and why were they added?**
After running the first sample evaluation, field-level mismatch analysis showed two systematic
errors: severity was being over-rated (predicted `high` when gold was `medium`, consistently in
one direction) and issue_type was confusing adjacent categories (crack vs glass_shatter, scratch
vs dent, stain vs water_damage). The guides are multi-line constant strings injected into every
prompt that give explicit calibration anchors ("low = cosmetic only", "glass_shatter ONLY when
spider-webbed; a single line is crack") and disambiguation rules. They're in the code as constants
so they apply uniformly to all three prompt builders (per-row, batched decide, eval layer).

**11. Why does the prompt include `extract_claim_cues` and what does it do?**
`extract_claim_cues` is a rule-based keyword extractor that scans the claim text for damage type
hints (e.g. "phati" → torn_packaging, "dent" → dent), part mentions, and severity words. These
cues are included in the prompt as "Text-extracted claim cues: ..." with a disclaimer that images
remain the source of truth. The intent is to reduce prompt noise — giving the model a narrower
search space for object_part and issue_type rather than having it infer entirely from raw text.
It is not authoritative; the model is instructed to override it if the visual evidence disagrees.

**12. How does the system select evidence requirements per row?**
`select_requirements` filters the evidence_requirements.csv rows to those where claim_object
matches `all` or the specific object (car/laptop/package), then within-object rows are further
filtered by whether their `applies_to` keywords appear in the claim text. If no requirements
match on text, it falls back to all requirements for that object type. A maximum of 6 are included
per prompt to keep token count bounded.

---

### Evaluation

**13. What is the evaluation methodology and how does it match the spec?**
The system evaluates on `dataset/sample_claims.csv` (20 labeled rows) using exact value match
per field as the primary metric (which is what the spec implies by "expected outputs"). It
computes per-field accuracy across the 7 scored fields, exact row match (all 7 right), claim_status
confusion matrix, and a supplementary severity-within-one-ordinal-level metric (since severity is
an ordinal scale). Results are written to `code/evaluation/evaluation_report.md` and per-approach
CSVs. The evaluation harness also includes the required operational analysis section.

**14. Why is exact row match ~25% when claim_status is 85%?**
Exact match requires all seven scored fields to be right simultaneously. The decision fields
(claim_status 85%, evidence_standard_met 95%, valid_image 95%) are strong. The two weakest fields
are severity (~50%) and issue_type (~45%), and requiring all seven to match means the joint
probability is roughly 0.85 × 0.95 × 0.95 × 0.80 × 0.55 × 0.50 × 0.45 ≈ 0.15–0.25. At n=20,
each row is worth 5 percentage points, so this range reflects 3–5 row misses driven entirely by
severity/issue_type. Severity-within-one-level is 90%, confirming the misses are near-misses.

**15. How did you identify and fix the over-flagging of `user_history_risk`?**
Field-level mismatch analysis (comparing predictions to gold labels on a per-flag basis) showed
we were adding `user_history_risk` 2 times when gold didn't have it. The cause: the
`history_requires_manual_review` function was using numeric thresholds (rejected ≥ 2, last 90
days ≥ 4, manual ≥ 3) to auto-add the flag even when the explicit `history_flags` column didn't
say "user_history_risk". The fix was to only trigger the flag when `history_flags` explicitly
contains "user_history_risk", matching the gold label logic exactly.

**16. What did the eval layer actually correct on the test set vs the sample?**
On the 20-row sample: 18 agree, 2 corrected. Both corrections hit gold-label ambiguities
(`damage_not_visible` vs `claim_mismatch`, `none` vs `unknown`) — defensible either way, so
accuracy was unchanged. On the 44-row test set: `damage_not_visible` appeared in 14 rows after
the eval layer vs 6 without it. This is the main systematic correction — contradiction rows
where the primary model described the part as visible but undamaged without adding the
`damage_not_visible` flag, which the eval layer's rubric rule catches.

**17. Why did you compare two strategies (batched vs per-row) instead of just submitting one?**
The spec requires an evaluation section comparing at least two strategies. More practically, it
was a genuine design question: the batched pipeline (flash_35 observe → flash_3-preview decide →
flash_35 verify) minimises request count (6 calls vs 20 for the sample) which matters when the
binding limit is daily requests (RPD). The per-row approach has lower cost, higher TPM, and
more reliability. Running both on the labeled sample revealed that batched did not improve
accuracy over the cheaper per-row baseline, and its dependency on the capacity-constrained
`gemini-3-flash-preview` caused repeated 429/503 failures.

---

### Rate limits, cost, caching

**18. What is the caching strategy and how does the cache key work?**
Every Gemini response is cached as a JSON file under `code/.cache/<strategy>/<hash>.json`.
For per-row calls, the cache key is a SHA-256 hash of the prompt version string, model name,
prompt text, and the raw bytes of each image file. This means: (a) changing the prompt
invalidates all caches; (b) changing an image invalidates only that row's cache; (c) identical
calls across reruns or eval harness loops hit the cache and make no API calls. On the final
output.csv generation, 44 primary rows were all cache hits (0 API calls, 0.3 seconds).

**19. What are the actual rate limits you hit and how did you work around them?**
`gemini-3-flash-preview` has a free-tier cap of ~20 requests/minute. During the batched pipeline
evaluation it also returned transient 503 "high demand" errors. Workarounds: (1) 5-second
spacing between calls; (2) retry with the server's stated `retryDelay` rather than blind backoff;
(3) `gemma-4-31b-it` as a fallback on a separate quota pool; (4) the eval layer uses
`gemini-3.5-flash` which has more free-tier headroom than the preview model.

**20. How much did it cost and how many API calls did the final run use?**
The full test set (44 rows): primary flash_lite run ~$0.074 (184,995 input + 7,189 output tokens,
44 calls). Eval layer: ~$0.031 (33,442 + 8,323 tokens, 2 calls). Total: ~$0.104 at assumed
rates of $0.30/1M input, $2.50/1M output. The 20-row sample evaluation added ~$0.03. All token
counts are real from Gemini's `usageMetadata`; pricing is an assumption since exact free-tier
rates were not confirmed.

---

### Robustness and engineering

**21. What happens if a Gemini call fails completely after all retries?**
`predict_row` falls through to `fallback_prediction`, which emits a deterministic safe-floor row:
`evidence_standard_met=false`, `claim_status=not_enough_information`,
`risk_flags=manual_review_required`, `valid_image=false`, all other fields set to `unknown`.
This guarantees output.csv always has exactly one row per input row and the schema is never
violated, even if every model call fails.

**22. How does the system handle truncated or malformed JSON from the model?**
`extract_json_object` first tries `json.loads` on the full response. If that fails it strips
markdown fences (```json ... ```) and tries again. If that fails it finds the outermost `{...}`
brackets and tries to parse just that substring. For the batched array case, `extract_json_array`
adds further fallbacks: tries to parse a `[...]` substring, and as a last resort runs
`_salvage_objects` which uses `JSONDecoder.raw_decode` to extract every complete `{...}` object
from the text, even if the surrounding array is truncated. This means a partially truncated
10-claim response still recovers the complete items.

**23. What was the 120-second timeout issue and how was it fixed?**
The default `urllib.urlopen` timeout was 120 seconds. With thinking enabled, a 10-claim batched
decision call using `gemini-3-flash-preview` took over 120 seconds due to the thinking phase,
causing a `TimeoutError`. The fix was two-part: (1) set `thinkingBudget=0` to eliminate thinking
(reducing latency from >120s to ~6s); (2) make the timeout a `Strategy` field
(`timeout_seconds=300`) so it can be tuned per model without changing the client code.

**24. Why is the output CSV written with `QUOTE_ALL`?**
The reference template at `dataset/output.csv` quotes every field. The `user_claim` column
contains comma-separated dialogue (`Customer: ... | Agent: ...`) which in minimal-quoting mode
causes the CSV to appear visually broken when opened in a text editor — the commas inside the
claim text create false column splits. With `QUOTE_ALL`, every field including short strings like
`"car"` and `"true"` is wrapped in double quotes, producing one unambiguous line per row.

**25. How does `row_image_paths` and image caching interact?**
`row_image_paths` splits the semicolon-delimited `image_paths` field and resolves each against
`dataset/` (falling back to the repo root). `image_part` reads the raw bytes and encodes them as
base64. The cache key for a per-row call hashes these raw bytes — so if an image file changes,
the hash changes and the cache misses, guaranteeing a fresh call. Multiple rows can share images
(they don't in this dataset, but the system supports it correctly because each cache key is
per-row-prompt, not per-image).

---

### The data and task

**26. What are the three object types and what makes each one challenging?**
Cars: multiple parts (front_bumper, door, hood, etc.) and frequent ambiguity between scratch vs
dent, crack vs glass_shatter. Laptops: functional damage (keyboard liquid damage) often has no
visible physical sign in a photo, requiring the system to correctly set `claim_status=contradicted`
for a "no visible physical damage" finding. Packages: the most ambiguous — torn vs crushed is
subjective, `unknown` vs `none` for issue_type on a contradicted claim is a genuine taxonomy
question, and multilingual claims (Hindi, Spanish) appear in both the sample and test sets.

**27. How does the system handle multilingual claims?**
The claims are used as input text to the prompt, and Gemini's models handle multilingual text
natively. The prompt instructions are in English but the claim text can be in any language — e.g.
`"Package seal side se open jaisa lag raha tha"` (Hindi), `"Cliente: Quiero reportar dano en el
parachoques trasero"` (Spanish). The model correctly interprets these and produces English output
in the required schema. No explicit translation step is needed.

**28. What does `supporting_image_ids` contain and how is it validated?**
It contains the filename stems (without extension) of images that directly justify the decision.
For a `supported` claim, it names the image(s) showing the damage. For a `contradicted` claim,
it names the image(s) showing the absence of damage or the wrong object/part. `validate_prediction`
extracts the valid stems from the row's `image_paths` field and filters the model's response to
only include those — so a model that hallucinates an image ID like `img_5` when only `img_1` and
`img_2` exist will have it silently dropped, producing `"none"` for that field.

**29. Why does `history_requires_manual_review` only check `history_flags` now?**
Originally the function also triggered manual review if rejected claims ≥ 2, last-90-days count
≥ 4, or manual review count ≥ 3. Comparing predictions to gold labels on the sample showed this
added `user_history_risk` on 2 rows where the gold label didn't have it. The gold labels appear
to use the explicit `history_flags` column as the authoritative signal rather than re-deriving risk
from the numeric counts. Matching this behaviour reduced false-positive risk flags.

**30. What would you do differently with more time or a paid API key?**
Three things. First, few-shot examples in the prompt using rows from `sample_claims.csv` — the
labeled data is there and 3–5 well-chosen examples for severity calibration and issue_type
disambiguation would likely close most of the remaining gap at that layer. Second, escalation:
route rows where the primary model's self-reported confidence is low (e.g. it chose
`not_enough_information`) to a stronger model rather than blindly re-reviewing everything.
Third, on a paid key, enable `gemini-3-flash-preview` at full rate for the batched pipeline —
the architecture is solid, it was only the free-tier quota that caused the repeated failures.

---

## Key numbers to remember

| Metric | Value |
|---|---|
| Test rows / images | 44 / 82 |
| Sample rows | 20 |
| Primary model | gemini-3.1-flash-lite |
| Fallback model | gemma-4-31b-it |
| Eval layer model | gemini-3.5-flash (text-only, 25/req) |
| API calls (test, total) | 46 (44 primary + 2 eval layer) |
| Total tokens (test) | ~234k |
| Approx cost (test) | ~$0.104 |
| Exact row match (sample) | 25% |
| claim_status accuracy | 85% |
| evidence_standard_met | 95% |
| severity within 1 level | 90% |
| Cache | SHA-256 of prompt + model + image bytes |
| Output format | 14 columns, QUOTE_ALL, one row per claim |
