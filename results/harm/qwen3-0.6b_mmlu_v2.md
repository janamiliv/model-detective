# Qwen3 0.6B harm on a fixed 1,000-question MMLU subset (greedy, CUDA, harm parser v2)

Reference: f16 run 1, accuracy 0.402 (99 unreadable answers). Null: a second greedy f16 run.
95% CIs by bootstrap over questions. Raw: `qwen3-0.6b_mmlu_v2.json`.

| Variant | Accuracy | Accuracy drop vs f16 [95% CI] | Flip rate [95% CI] | Correct→wrong | Wrong→correct | Unreadable |
|---|---|---|---|---|---|---|
| f16 (null, run 2) | 0.405 | −0.003 [−0.009, +0.003] | 0.009 [0.004, 0.015] | 3 | 6 | 99 |
| Q8_0 | 0.408 | −0.006 [−0.017, +0.006] | **0.034** [0.023, 0.045] | 14 | 20 | 117 |
| Q6_K | 0.421 | −0.019 [−0.035, −0.003] | **0.081** [0.065, 0.097] | 31 | 50 | 65 |
| Q5_K_M | 0.368 | **+0.034** [+0.012, +0.054] | **0.132** [0.110, 0.152] | 83 | 49 | 254 |
| Q4_K_M | 0.346 | **+0.056** [+0.032, +0.082] | **0.194** [0.170, 0.218] | 125 | 69 | 180 |
| Q3_K_M | 0.341 | **+0.061** [+0.029, +0.094] | **0.283** [0.257, 0.311] | 172 | 111 | 113 |
| Q2_K | 0.121 | **+0.281** [+0.244, +0.318] | **0.411** [0.380, 0.441] | 346 | 65 | 518 |

Bold: significant under the pre-registered rule (prereg §10): flip-rate CI excludes the null's
flip rate, or the accuracy-drop CI excludes zero (in the harmful direction).

## Notes

- **The null flip rate is not zero (0.9%).** Two greedy f16 runs on CUDA disagree on 9 of 1,000
  questions. This is batched-GPU non-determinism (8 parallel slots), the pitfall PLAN.md §2 lists
  as #5. It is why the null run exists; every variant's flip rate is compared against it.
- **Q8_0 already flips 3.4% of answers**, significantly above the null, while accuracy is
  unchanged: it moves answers in both directions (14 lost, 20 gained). This is the "accuracy flat,
  answers flip" pattern the harm-side literature describes.
- **Q6_K's accuracy goes up** (0.421 vs 0.402, CI excludes zero in the beneficial direction). Not
  harm in the accuracy sense, but 8.1% of answers change.
- **Unreadable answers vary by variant** (65–518). They count as wrong, so part of the accuracy
  drop at Q5_K_M and Q2_K is format breakage rather than wrong knowledge. Reported, not corrected.

## KL divergence of next-token distributions vs f16

`llama-perplexity --kl-divergence`, 16 chunks × 512 tokens of WikiText-2 test (`configs/kl_text.txt`),
CUDA. Raw: `qwen3-0.6b_kl.json`.

| Variant | Mean KL (nats) ± SE | Median KL | 99th pct KL | Same top-1 token |
|---|---|---|---|---|
| f16 (self, null) | −0.000015 (rounding) | −0.000014 | 0.000 | 100.0% |
| Q8_0 | 0.0032 ± 0.0001 | 0.0025 | 0.017 | 96.5% |
| Q6_K | 0.0135 ± 0.0003 | 0.0100 | 0.073 | 93.4% |
| Q5_K_M | 0.0376 ± 0.0010 | 0.0252 | 0.217 | 89.6% |
| Q4_K_M | 0.1130 ± 0.0025 | 0.0766 | 0.728 | 82.1% |
| Q3_K_M | 0.4370 ± 0.0080 | 0.3007 | 2.474 | 66.9% |
| Q2_K | 2.9091 ± 0.0330 | 2.4776 | 9.605 | 28.4% |

KL rises monotonically with falling bit-width and tracks the flip rate. On the probe, detection
starts between Q6_K (KL 0.013) and Q5_K_M (KL 0.038): at this size the 16-bit-string detector
needs a mean next-token KL of roughly 0.02–0.04 nats before it can see the change within 125
queries.
