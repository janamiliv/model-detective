# Qwen3 1.7B harm (fixed 1,000-question MMLU subset, greedy, CPU, harm parser v2; core cells)

Reference: f16 run 1, accuracy 0.544 (38 unreadable). Raw: `qwen3-1.7b_mmlu_v2.json`, `qwen3-1.7b_kl.json`.

| Variant | Accuracy | Accuracy drop [95% CI] | Flip rate [95% CI] | C→W | W→C | Unreadable | Mean KL (nats) | Same top-1 |
|---|---|---|---|---|---|---|---|---|
| f16 (null, run 2) | 0.544 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0 | 0 | 38 | 0.0000 | 100.0% |
| Q8_0 | 0.544 | 0.000 [−0.009, +0.009] | **0.022** [0.013, 0.032] | 11 | 11 | 47 | 0.0034 | 97.1% |
| Q4_K_M | 0.562 | −0.018 [−0.039, +0.003] | **0.118** [0.098, 0.139] | 50 | 68 | 1 | 0.1048 | 86.3% |
| Q2_K | 0.134 | **+0.410** [+0.372, +0.448] | **0.532** [0.502, 0.563] | 471 | 61 | 547 | 2.4639 | 44.6% |

## Notes

- **Greedy decoding on CPU is deterministic**: the two f16 runs agree on all 1,000 answers (on
  CUDA with 8 slots at 0.6B they disagreed on 0.9%). The null flip rate is exactly zero here.
- **Q8_0 changes 2.2% of answers** in both directions with unchanged accuracy, the same pattern as
  at 0.6B (3.4%).
- **Q4_K_M's accuracy is not lower** (0.562 vs 0.544, CI includes zero), and it almost never fails
  to give a readable letter (1 vs 38 for f16), yet 11.8% of answers differ from f16.
- KL to f16 is nearly identical to 0.6B at matched bit-width (Q8_0 0.0034 vs 0.0032; Q4_K_M 0.105
  vs 0.113): per-token distortion does not shrink with size here, while the probe's output
  distribution does (mode collapse).

## H2 at 1.7B

Harm (flip rate) and detectability both appear at the mildest variant tested, Q8_0 (2.2% flips;
detected at 32 queries on content-only features). With only Q8_0, Q4_K_M and Q2_K collected at this
size (core-cell scope), the ordering between the onset of harm and of detectability cannot be
resolved more finely: at 1.7B neither precedes the other within the tested grid.
