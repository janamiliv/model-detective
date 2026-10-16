# Qwen3 0.6B, primary probe (16 random bits): detection power by bit-width

Pre-registered protocol (docs/prereg.md §9, tag prereg-v1, with dated deviations): random forest
on content-only visible-string features, 20 random splits, f16 reference split 50/25/25, f16 null
pool (independent run) only scored, 1,000 episodes per budget, alpha = 0.05. CPU backend,
llama.cpp b11429. 2,000 responses per pool. Raw numbers: `qwen3-0.6b__short_bits_16.json`.

## Headline (content-only features, primary)

| Variant | bits/weight | Power, 1 query | Power, 16 queries | Power, 64 queries | Queries to 80% power | Queries to 95% power | Notes |
|---|---|---|---|---|---|---|---|
| Q8_0 | 8.6 | 0.05 | 0.05 | 0.06 | > 125 | > 125 | indistinguishable from the null |
| Q6_K | 6.6 | 0.05 | 0.09 | 0.17 | > 125 | > 125 | faint signal only at the largest budgets |
| Q5_K_M | 5.9 | 0.16 | 0.94 | 0.98 | **16** | **64** | |
| Q4_K_M | 5.2 | 0.52 | 0.92 | 1.00 | **4** | **32** | |
| Q3_K_M | 4.4 | 0.37 | 0.99 | 1.00 | **2** | **4** | |
| Q2_K | 3.7 | 0.99 | 1.00 | 1.00 | 1 | 1 | **degenerate** (parse success 0.6%): detecting broken output is trivial |

"> 125": power never reached the target at any reliable budget. Reliable budgets are m ≤ 125
(calibration split of 500 responses ÷ 4) with the null false-positive rate in [0.02, 0.08]; m = 128
and 256 are shown in the JSON but flagged unreliable, as pre-registered.

## Controls

- **Null (f16 run 2 vs reference):** false-positive rate 0.029–0.060 across all reliable budgets
  and all six comparisons. At m = 128 and 256 it rises to 0.06–0.27, because those episodes
  reuse most of the 500-response calibration split; this is exactly what the reliability rule
  excludes.
- **Content-only vs full features:** full features give the same queries-to-detect at Q3_K_M and
  Q5_K_M (80%) and are faster at Q4_K_M (2 vs 4); single-query power is 0.04–0.09 higher with
  full features. Length and format carry some extra signal; the headline uses content-only.

## Reading against the hypotheses (0.6B only; 1.7B and 4B pending)

- **H1 (bit-width curve):** supported at 0.6B. Q8_0 is not detectable within 100 queries (power
  0.06 at 64); Q4_K_M and below are (80% power at ≤ 4 queries). Queries-to-detect falls
  monotonically with bit-width: > 125, > 125, 16, 4, 2, 1.
- **Cross-check with IRIS:** IRIS reported a per-query tell rate of 0.35–0.42 for Ollama's q4 vs
  fp16 at 0.6B; our single-query power for Q4_K_M is 0.52 (content-only), consistent given the
  larger training pools here (1,000 vs 100 per side).

## Caveats

- Power is not strictly monotone in m for Q4_K_M (0.95 at 8, 0.92 at 16) and Q5_K_M (0.94 at 16,
  0.81 at 32). The episode statistic is the mean of −log P(reference), which is heavy-tailed (one
  confident tell dominates a short episode). This is the pre-registered statistic and is reported
  as is; a robust statistic (median or trimmed mean) is an exploratory follow-up, labeled as such.
- Single-query power is higher for Q4_K_M (0.52) than Q3_K_M (0.37), while Q3_K_M needs fewer
  queries for 80%. Q3_K_M shifts the whole distribution a little (many weak tells); Q4_K_M
  produces fewer but more distinctive outputs (its top string is 24% of responses).
