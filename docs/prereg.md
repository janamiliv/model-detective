# Pre-registration — DRAFT for review, not yet frozen

Status: drafted 2026-10-14 after the Phase 1 baseline reproduction. To be reviewed by the project
owner and then frozen by a commit tagged `prereg-v1` before any Phase 3 collection beyond the
Phase 1 cells. Any change after the freeze goes in the dated "Deviations" section at the end.

## 1. Question

How many black-box text queries are needed to distinguish a GGUF-quantized Qwen3 model from its
f16 original, as a function of bit-width, and does that number fall below the point where
quantization measurably harms task behavior?

## 2. Hypotheses (unchanged from PLAN.md Section 3)

- **H1 (bit-width curve).** Detectability rises monotonically as bit-width falls. Q8_0 is not
  detectable within 100 queries on the primary probe at 80% power; Q4_K_M and below are.
- **H2 (detection-harm gap).** Quantization becomes detectable (queries-to-detect at 80% power
  ≤ 100) at a higher bit-width than the one where harm becomes significant (defined in §5).
  The reverse ordering is an equally reportable outcome.
- **H3 (method fingerprint, stretch).** At matched nominal bit-width, GGUF K-quants and a second
  quantization method are distinguishable by a classifier above a shuffled-label null.
- **H4 (scale).** Queries-to-detect at a fixed bit-width decreases with model size
  (0.6B → 1.7B → 4B).
- **H5 (human vs classifier, demo).** Human guesses from short outputs are near chance while the
  classifier is not. (Measured only if the demo collects anonymous aggregates.)

## 3. Models and variants

- Qwen3 0.6B and 1.7B locally; 4B if feasible on CPU, else cloud with approval. HF revisions
  pinned in `configs/models.yaml`.
- Reference: f16 GGUF converted with llama.cpp b11429 (`convert_hf_to_gguf.py --outtype f16`).
- Variants: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K via `llama-quantize` b11429. No importance
  matrix (plain K-quants) so that the variants differ only in bit-width.
- Null control: a second, independent f16 run (fresh server process, disjoint seeds) in every
  comparison.

## 4. Runtime and sampler (fixed for all cells)

- llama-server build b11429 / d812350, one machine (docs/hardware.md), CPU unless the GPU gate is
  re-passed after a driver update, in which case **all** cells are re-collected on the GPU; no
  mixing of backends within a comparison.
- Sampler: temperature 1.0, top_p 1.0, top_k 0, min_p 0, repeat_penalty 1.0. Qwen3 thinking off.
- Seeds: unique per request, derived deterministically from (run_id, probe, index); recorded.
- 8 server slots, 8 threads.

## 5. Probes

- **Primary probe (decides H1, H2, H4):** IRIS `short_bits_16` ("Generate exactly 16 symbols by
  independently choosing uniformly between 0 and 1. Output the sequence only, with no spaces and
  no explanation."), no system prompt, 88-token cap.
- Secondary probes (reported, never used to pick the headline): IRIS `short_digits_8`,
  `short_bits_32`, `single_number_100` (with the corrected answer parser, §7); the One Token
  battery (10 tasks × en/ru/zh/ar, verbatim system prompts, 16-token cap); MET-style natural
  prompts (25 short continuation prompts, 50-token cap) for the Hamming-MMD detector.
- Optional extension, clearly labeled: One Token tasks in Serbian (Latin and Cyrillic).

## 6. Sample sizes

- Primary probe: 2,000 responses per (model, variant), plus 2 × 2,000 f16 (reference and null).
- Other IRIS probes: 1,000 per cell. One Token: 1,000 per (model, variant, task, language) for
  en; 300 for ru/zh/ar. MET prompts: 40 per prompt per variant (1,000 per variant).
- Logprobs: top-20 next-token logprobs recorded for every one-token cell (oracle distance).

## 7. Parsing and exclusion rules

- IRIS char parser as implemented in `probes/iris.py`. Number parser: the **last** integer in the
  response, and the row is flagged `echo` if the only integers present also appear in the prompt.
- No response is dropped. Rows with request errors are counted and reported; they are excluded
  from feature extraction only if the error is a transport failure (not a model output).
- A cell is excluded from headline analysis only if its parse-success rate is below 50%; this is
  reported, never silent.

## 8. Detectors

- (a) Jensen-Shannon divergence (base 2) between one-token answer distributions, mean over cells
  with ≥ 10 valid answers (One Token).
- (b) MMD with Hamming kernel, permutation p-values (MET); cross-checked against the
  `model-equality-testing` package.
- (c) Random-forest tell rate on visible-string features (IRIS), **content-only features as the
  primary set**, full features secondary. Tell-rate protocol as in `detect/tell_rate.py`.
- (d) Optional: rank-uniformity test on local logits.

## 9. Primary metric and statistics

- **Queries-to-detect**: smallest m in {1, 2, 4, 8, 16, 32, 64, 128, 256} at which power ≥ 0.80
  (and ≥ 0.95) at alpha = 0.05, where an episode is m responses, the decision statistic is the
  mean per-response score, the threshold is calibrated on a disjoint null split, and power is
  measured on held-out data. Permutation or exact calibration; no normal approximation without a
  check. 1,000 bootstrap resamples for 95% CIs.
- Controls on every plot: f16-vs-f16 null (FPR must be within [0.02, 0.08] at nominal 0.05),
  and the content-only versus full comparison.
- Secondary: single-query AUROC, equal error rate at fixed m.

## 10. Harm metrics (H2)

- Fixed 1,000-question subset of MMLU (seed 20261014), greedy decoding, identical prompts.
- Accuracy, flip rate (correct→wrong plus wrong→correct), and KL divergence of next-token
  distributions vs f16 on a fixed held-out text sample via `llama-perplexity --kl-divergence`.
- "Harm becomes significant" = flip rate's 95% bootstrap CI excludes the f16-vs-f16 null flip
  rate, or accuracy drop's CI excludes zero.

## 11. What we will not do

Probe external APIs; build evasion or spoofing tools; choose the headline probe after seeing
results; truncate axes.

## Deviations (dated)

None yet.
