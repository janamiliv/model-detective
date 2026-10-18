# Detector (a): One Token JSD fingerprint (English cells)

Battery: the ten paper-1 tasks of Bruckner (2026), verbatim prompts and system prompt, English,
300 responses per cell per pool, max 16 tokens, temperature 1. Distance: mean over cells of
base-2 JSD between normalized valid answers; cells need ≥ 10 valid answers in every split part
(the paper's MIN_N). Episode = m answers per usable cell; total queries = cells × m.
Threshold from a disjoint calibration split; 20 splits. Raw: `qwen3-0.6b.json`, `qwen3-1.7b.json`.

## Qwen3 1.7B (CPU): the battery works

| Pool | Battery JSD vs f16 run 1 | Valid answers | Power at 10 / 20 / 40 / 80 / 160 queries | Queries to 80% |
|---|---|---|---|---|
| f16 run 2 (null) | 0.023 | 100% | false-positive rate 0.034–0.041 | |
| Q8_0 | 0.043 | 100% | 0.10 / 0.14 / 0.16 / 0.26 / 0.46 | > 160 |
| Q4_K_M | 0.139 | 100% | 0.13 / 0.31 / 0.72 / **1.00** / 1.00 | **80** |
| Q2_K | 0.999 | 70% | 1.00 at every budget (7 usable cells) | 7 (degenerate output) |

Compared with detector (c) on 16 random bits at the same size (Q8_0 32 queries, Q4_K_M 8), the
one-word battery needs about ten times more queries for Q4_K_M and does not reach 80% for Q8_0
within 160. This matches the literature's warning that low-entropy probes are weak detectors of
quantization (PLAN.md §2, pitfall 3; IRIS's `single_number_100` result).

## Qwen3 0.6B (CUDA): not applicable — the reference fails the task

Under the battery's one-word system prompt the f16 model answers **"no"** to 98–100% of every
question ("Name a random number between 1 and 100." → "no"); Q8_0, Q6_K and Q5_K_M do the same.
Q4_K_M and Q3_K_M answer **"1"** to nearly every question (including "Name a random color"); Q2_K
produces prose. Only the seven open-ended cells pass the validity filter, because "no" and "1" are
"valid" single words there.

So at 0.6B this detector measures a switch between two failure modes ("no" vs "1"): Q4_K_M,
Q3_K_M and Q2_K are trivially separated (7 queries, power 1.0) and Q8_0–Q5_K_M are not separated
at all (power ≤ 0.02). This is reported for completeness and is **not** used as evidence about
quantization detectability: the reference model does not perform the task.
