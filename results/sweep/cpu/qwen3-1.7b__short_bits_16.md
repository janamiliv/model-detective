# Qwen3 1.7B, primary probe (16 random bits): detection power (core cells)

Same pre-registered protocol as 0.6B; core-cell scope (prereg deviation 2026-10-18): f16 ×2,
Q8_0, Q4_K_M, Q2_K; 2,000 responses per pool; CPU, llama.cpp b11429. Raw: `qwen3-1.7b__short_bits_16.json`.

## The 1.7B model is mode-collapsed on this probe

Asked for 16 random bits at temperature 1 with no truncation, the f16 model produces about 15
distinct strings in 2,000 responses:

| Pool | Top answers (share of responses) | Distinct strings |
|---|---|---|
| f16 run 1 | `0000000000000000` 55.1%, `01010101010101010101` 26.6%, `000000000000000000` 14.4% | 0.8% |
| f16 run 2 | `0000000000000000` 54.7%, `01010101010101010101` 26.9%, `000000000000000000` 13.7% | 0.8% |
| Q8_0 | `000000000000000000` 50.0%, `0000000000000000` 42.1%, `01010101010101010101` 5.9% | 0.6% |
| Q4_K_M | `01010101010101010101` 35.5%, `000000000000000000` 31.9%, `010` repeated (88 chars) 18.5% | 2.4% |
| Q2_K | parse success 0%: output is not a bit string | 75% |

Quantization here does not add randomness; it **moves probability mass between a few modal
strings** (Q8_0 swaps 16 zeros for 18 zeros as the favourite).

## Detection (content-only features, primary)

| Variant | Power, 1 query | Power, 16 | Queries to 80% | Queries to 95% | 0.6B, queries to 80% |
|---|---|---|---|---|---|
| Q8_0 | 0.24 (unreliable budget) | 0.71 | **32** | 32 | > 125 |
| Q4_K_M | 0.33 | 0.98 | **8** | 16 | 4 |
| Q2_K | 0.99 (unreliable budget) | 1.00 | 8 (only m = 8 reliable) | 8 | 1 |

Full features: Q8_0 8, Q4_K_M 4, Q2_K 4 queries to 80%. The extra speed comes from length: the
Q8_0 shift (16 → 18 zeros) is a pure length change, which content-only features ignore by design.

## Caveats (reported, not corrected)

1. **Degenerate flag on Q8_0 is a rule artifact.** The pre-registered rule (deviation 1) flags a
   pool when one string exceeds 50% of responses. The f16 reference itself is at 55%; Q8_0 sits at
   50.05% and parses 100%. The rule was meant for broken output; at 1.7B it fires on legitimate
   mode collapse. The flag stays in the JSON; this note explains it. Q2_K's flag is genuine.
2. **Null FPR out of band at small budgets** (0.10–0.24 at m = 1–2) because scores take only a
   few distinct values when outputs collapse; the threshold quantile lands on a tie. The
   pre-registered reliability rule excludes those budgets; queries-to-detect uses reliable ones only.
3. **H4 is mixed at this point.** Q8_0 becomes detectable at 1.7B (supports H4); Q4_K_M needs
   slightly more queries than at 0.6B (does not). IRIS reported q4 at 1.7B as much easier than at
   0.6B on Ollama; their sampler (Ollama default top-k) and runtime differ. Mode collapse is a
   plausible mechanism (less information per answer, but mode shifts are easy to see); 4B will
   tell whether the trend continues.
