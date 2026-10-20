# Qwen3 4B, primary probe (16 random bits): detection power (core cells)

Same pre-registered protocol; core-cell scope (prereg deviation 2026-10-18); CPU, llama.cpp b11429;
2,000 responses per pool. Raw: `qwen3-4b__short_bits_16.json`.

## The 4B model follows the instruction but repeats a few "random-looking" strings

All pools parse at 100% and produce exactly 16 bits (17 tokens). f16 uses about 48 distinct
strings in 2,000 responses; its top three (`0101100010110011` 25.5%, `0101100010110101` 20.2%,
`0101100100110110` 12.6%) cover 58%. Q8_0 keeps the same top three with slightly flatter shares;
Q4_K_M promotes different strings (`0101100100110110` 20.1%, `0101100101101010` 19.7%).
Q2_K still parses at 100% at this size (unlike 0.6B and 1.7B) but answers with long strings
(mean 88 tokens) and is not flagged degenerate.

## Detection (content-only features, primary)

| Variant | Power, 1 query | Power, 16 | Power, 64 | Queries to 80% | Queries to 95% |
|---|---|---|---|---|---|
| Q8_0 | 0.05 | 0.11 | 0.22 | > 125 | > 125 |
| Q4_K_M | 0.74 | 1.00 | 1.00 | **2** | **4** |
| Q2_K | 1.00 (m = 1 unreliable) | 1.00 | 1.00 | 2 | 2 |

Null false-positive rate 0.03–0.06 at reliable budgets. Full features give the same numbers for
Q8_0 and Q4_K_M.
