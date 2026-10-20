# Detection across model sizes (H4): primary probe, content-only features

| | 0.6B | 1.7B | 4B |
|---|---|---|---|
| Output on "16 random bits" (f16) | varied, often too long (79% distinct strings) | mode-collapsed (~15 strings, top 55%) | exact 16 bits, ~48 strings, top 26% |
| **Q8_0**: power at 1 query | 0.05 | 0.24 (unreliable budget) | 0.05 |
| **Q8_0**: queries to 80% | > 125 | **32** | > 125 |
| **Q4_K_M**: power at 1 query | 0.52 | 0.33 | **0.74** |
| **Q4_K_M**: queries to 80% | 4 | 8 | **2** |
| Q2_K | degenerate | degenerate | detected at 2 queries, not degenerate |
| One Token battery, Q4_K_M queries to 80% | not applicable (reference answers "no") | 80 | 80 |
| One Token battery, Q8_0 | not applicable | > 160 | > 160 |

**H4 (detectability rises with size): partly supported.** From 0.6B to 4B, Q4_K_M gets easier
(single-query power 0.52 → 0.74, queries 4 → 2), in the direction IRIS reported (0.35 → 1.00 tell
rate for Ollama q4 at 0.6B → 4B). It is **not monotone**: 1.7B is harder for Q4_K_M and easier for
Q8_0 than either neighbour, because at 1.7B the model collapses onto a handful of strings and
quantization shows up as a shift between those modes. Q8_0 is undetectable within 125 queries at
both 0.6B and 4B. The size effect is therefore driven as much by how each model "randomizes" as
by size itself; three sizes of one family cannot separate the two.

The low-entropy One Token battery is about 10–40× less query-efficient than the 16-bit probe at
both sizes where it works, and never detects Q8_0 within 160 queries.
