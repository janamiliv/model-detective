# Related work: what the primary sources actually did

Written 2026-10-14 from full-text reads of the three primary papers and their code. Every number
below was checked against the paper text or shipped data; items we could not verify are marked
UNVERIFIED. PLAN.md Section 2 was a web-research summary; where it disagreed with the sources,
this file wins and the disagreement is noted.

---

## 1. IRIS (Zhang, Zhang, Qin, 2026) — closest prior work

**Which Model Is Actually Serving You? IRIS: Budgeted Black-Box Auditing of Model Substitution and
Routing Dilution in LLM Gateways.** Yuewei Zhang, Zhi-Hai Zhang (Tsinghua), Hanzhang Qin (NUS).
arXiv 2607.20860v1, 23 Jul 2026. Code and data: github.com/Photen/IRIS-audit (commit 3d859ab;
code PolyForm Noncommercial 1.0.0, data CC BY-NC 4.0).

**Summary.** IRIS audits whether an LLM endpoint serves the advertised model using only returned
text. It sends fixed random-generation prompts (a digit, an integer 1–100, a coin flip, or 8/16/32-
symbol bit and digit strings), maps each visible response to a 179-dimensional vector of format,
symbol-frequency, run, transition and positional features, and trains a random forest over enrolled
endpoints. The per-response score is s(y) = −log P̂(reference | y). Whole-stream substitution is
detected from the mean score over m queries; fractional routing ("dilution", a fraction ε of
requests sent to a cheaper model) is detected by counting response-level *tells* (s(y) > τ with τ a
reference quantile) with a binomial test. Quantization is one appendix stress test (S10.1), not
the paper's main subject. Important: the paper is a general substitution auditor; our project
extends only its q4-vs-fp16 appendix.

**Setup details we copy.**

| Item | IRIS |
|---|---|
| Probes (verbatim) | `short_bits_16`: "Generate exactly 16 symbols by independently choosing uniformly between 0 and 1. Output the sequence only, with no spaces and no explanation." `single_number_100`: "Choose one uniformly random integer between 1 and 100 inclusive. Output only the integer." Others in `src/modeldetective/probes/iris.py`. |
| System prompt | none; single user turn |
| Decoding | temperature 1.0, top_p 1.0; **top_k never set** (Ollama default applied, value unrecorded); no seed; thinking off (`think: false`) |
| Decode cap | 48 tokens for single draws; max(72, 3L+40) for strings (88 for 16 bits) |
| Runtime | Ollama 0.12.5 on A40 GPUs; `qwen3:0.6b-fp16` vs `qwen3:0.6b` (Ollama default tag, called "q4"; exact GGUF type UNVERIFIED, likely Q4_K_M) |
| Parser | char parser keeps every alphabet character anywhere in the response, in order; number parser takes the first integer; parse_success = parsed_len > 0 and invalid_rate ≤ 0.25; failed parses are kept as rows |
| Features | 179 total; "content-only"/length-ablated set drops length, format, counts, runs, compression (the two quant scripts' ablation sets differ slightly; we follow `quant_twin.py --ablate-length`) |
| Classifier | RandomForest, 300 trees, sqrt features, balanced classes, seed 20260617 |
| Protocol | 40 stratified 50/50 splits; τ = 0.95 quantile of held-out reference scores; q = fraction of held-out substitute scores > τ; report mean q and 2.5/97.5 percentiles across splits; "clears floor" if the 2.5th percentile > α |
| Samples | Table 8: 200 per label per probe (400 for `short_digits_8` and `single_number_100` in the raw file); negative control: an independent second fp16 run, 250 per probe |

**Quantization results (App. S10.1, Table 8, 0.6B, α ≈ 0.05).**

| Probe | Features | q (tell rate) | 95% band | Detect | m* (ε=0.1) |
|---|---|---|---|---|---|
| c_{2,16} 16 bits | full (52) | 0.42 | [0.27, 0.52] | yes | 72 |
| c_{2,16} 16 bits | length-ablated (28) | 0.37 | [0.17, 0.51] | yes | 82 |
| c_{100,1} number 1–100 | full (165) | 0.06 | [0.02, 0.11] | no | |
| c_{100,1} number 1–100 | length-ablated (132) | 0.00 | [0.00, 0.00] | no | "a length artifact" |
| fp16 vs fp16 null, 16 bits | ablated | 0.03 | [0.01, 0.07] | no | |

Across sizes (text only, backed by `results/quant_breadth_metrics.json`, length-ablated, 16 bits,
5-fold out-of-fold): q = 0.35 at 0.6B (AUROC 0.72), 0.74 at 1.7B (AUROC 0.95), 1.00 at 4B
(AUROC 1.00); m* falls from about 86 to about 30. Authors' own framing: "detectability is probe-
and model-dependent ... we do not claim a clean entropy law for quantization."

**Confounds the authors flag.** Provider and precision are perfectly confounded in their live
OpenRouter audit, so that result shows endpoint-configuration identifiability, not precision.
The same nominal model on local Ollama vs a gateway is itself distinguishable (S7), so the
reference must be collected under the same serving stack. Low-entropy probes carry a length
artifact. Certifying small FPRs needs thousands of reference samples.

**Where PLAN.md was imprecise.** The 0.35/0.74/1.00 numbers are from the across-sizes paragraph
(length-ablated), not Table 8 (which gives 0.37 ablated / 0.42 full at 0.6B with a different
split protocol). Both are now recorded.

**Reproduction notes.** The shipped collectors talk only to Ollama's `/api/generate`; nothing
targets an OpenAI-compatible server. We therefore reimplemented probes, parser, features and the
tell-rate protocol in `src/modeldetective/` rather than patching their scripts. Their repo files
check out with CRLF and some scripts call `open()` without an encoding; irrelevant to us since we
only read their data and results for comparison.

---

## 2. One Token Is Enough (Bruckner, 2026) — probe battery and JSD fingerprint

**One Token Is Enough: Fingerprinting and Verifying Large Language Models from Single-Token Output
Distributions.** Tomáš Bruckner (Prague University of Economics and Business). arXiv 2607.10252v1,
11 Jul 2026. Author's code (PAMELA) Zenodo 10.5281/zenodo.21278793 (MIT); data Zenodo
10.5281/zenodo.21278557 (CC BY 4.0). The GitHub repo linked in PLAN.md
(ToseaAI/llm-fingerprint-detector) is an **independent third-party reimplementation** with a
different battery (8 tasks × en/zh, paraphrases, different system prompt), not the paper's code.

**Summary.** The fingerprint is the empirical distribution of one-word answers to trivial prompts
("Name a random number between 1 and 100.") under a fixed one-word system prompt, at temperature
1.0. The battery is 10 tasks × 4 languages (English, Russian, Chinese, Arabic) = 40 cells, 30
repetitions per cell. Two models are compared by the mean Jensen-Shannon divergence (base 2, not
square-rooted) over cells where both have at least 10 valid answers. In a census of 165 chat
models via OpenRouter (326,047 responses, $34.44), split halves of the same model sit at median
JSD 0.140 versus 0.463 for different models; leave-one-out 1-NN recovers the documented family
with 59.5% accuracy against an 18.4% majority-class rate; split-half verification reaches AUC
0.971 and EER 7.3% with all 40 cells, EER 10.6% (band 8–14%) with 8 random cells, i.e. about 120
queries at 15 answers per cell. The same model served by different providers has median distance
0.227, and 10 of 34 provider pairs diverge beyond the impostor range.

**Setup details.**

| Item | One Token |
|---|---|
| System prompt (en, verbatim) | "Answer with a single word or a single number only. Output nothing else: no explanation, no punctuation, no quotation marks." (ru/zh/ar versions in `configs/one_token_prompts.zenodo.json`) |
| Tasks | random number 1–100, 1–10, favorite number, random letter, random word, random color, favorite color, random animal, random city, coin flip |
| Decoding | temperature 1.0; top_p and top_k provider defaults; no seed; `max_tokens` **16** (not 1; "one token" is nominal, valid answers are 1–6 tokens) |
| Parsing | NFC, strip punctuation, case fold, map Arabic-Indic and Chinese numerals, first token, color lexicon; classes valid / invalid / refusal / empty / post-reasoning; nothing dropped silently; overall validity 97.6% |
| Distance | mean over cells of JSD(p̂_A, p̂_B), base 2, MIN_N = 10 valid per cell |
| Samples | 30 reps per cell at T=1 (15 for frontier-priced models), 3 at T=0 |

**What it does not do.** It never compares a model to its own quantized variant; the catalog has
no quantization field. Its "same weights, different provider" distances (median 0.227, up to
0.716) already overlap the impostor range, so this metric is untested for quantization and the
paper offers no calibration of JSD against precision. Where PLAN.md said "1 output token per
query" and "8 probe cells", the facts are: max_tokens 16, and the 8-cell EER is a mean over
random 8-cell subsets, not a curated subset.

**Use in our project.** The verbatim battery (prompts and system prompts, four languages) becomes
our one-token probe set in Phase 3 and detector (a) in Phase 4 is its JSD. We will report JSD
between fp16 and each quantized variant against the split-half null, which is the measurement
this paper did not make.

---

## 3. Model Equality Testing (Gao, Liang, Guestrin, ICLR 2025) — two-sample baseline

**Model Equality Testing: Which Model Is This API Serving?** Irena Gao, Percy Liang, Carlos
Guestrin (Stanford). ICLR 2025 poster; arXiv 2410.20247v2 (8 Apr 2025). Code
github.com/i-gao/model-equality-testing (MIT), package `model-equality-testing` 0.0.2 on PyPI.

**Summary.** Frames API auditing as a two-sample test between prompt-completion pairs from the
API and from a reference (released weights run locally at fp32). Uses kernel MMD with a Hamming
kernel, k(y, y') = number of positions where right-padded completions agree, which is far more
sample-efficient than one-hot or all-substrings kernels. On a 25-prompt multilingual Wikipedia
continuation task (L = 50 tokens, temperature 1, no top-k/top-p, α = 0.05) it reaches a median
power of 77.4% across fp16/int8/nf4/watermark alternatives with 10 samples per prompt (250
total). Applied to 31 commercial Llama-3/3.1 endpoints in summer 2024, 11 were flagged as
differing from Meta's reference weights at under $1 per audit.

**Setup details.**

| Item | MET |
|---|---|
| Prompts | 25 random 100-character strings from en/de/es/fr/ru Wikipedia with "Continue the paragraph. Do not output anything except the continuation..." |
| Decoding | temperature 1, top_k 0, top_p 1.0, max_new_tokens 50 (Wikipedia) or 250 (HumanEval, UltraChat) |
| Quantization | Transformers bitsandbytes: int8 (LLM.int8) and nf4; fp16 cast. **No GGUF.** |
| Statistic | unbiased MMD U-statistic with Hamming kernel; p-values by simulated null or permutation (100–1000 permutations) |
| Samples | N = 10 per prompt for headline results; dataset has 15,000/prompt fp32, 5,000/prompt for other local variants |

**Quantization power (Table 8, token space, N = 10m).** nf4 is consistently detectable on Llama
and OLMo (0.89–1.00); int8 is model-dependent: Llama-3 8B 0.79, Llama-3.1 8B **0.08**, Mistral 7B
0.30, 70B models 1.00; Gemma-2 9B nf4 0.12 / int8 0.06. fp16 vs fp32 is the hardest alternative
("differ in ways not captured by the Hamming kernel"). In character space (what an API shows) power
drops and must be bought back with samples: Llama-3 8B int8 0.30 at N = 10m, 0.96 at N = 50m.

**Package use.** `run_two_sample_test(s1, s2, pvalue_type="permutation_pvalue",
stat_type="mmd_hamming")` on `CompletionSample`s built with `tokenize_unicode` and
`pad_to_length`. Note the default `stat_type` is *not* the paper's test. The package's kernel
normalization differs cosmetically from the paper's eq. (4). It is installed here as the
`baseline` extra and will cross-check our own MMD implementation in Phase 4.

**Limitations relevant to us.** The Hamming MMD is a pseudo-metric (sees only per-position
marginals). The dataset's license is not stated. Jailbreaks and system-prompt distortions are not
studied, contrary to the impression PLAN.md's table might give.

---

## 4. Other work (not read in full; cite from abstracts only)

- Zhu et al., Rank-Based Uniformity Test (arXiv 2506.06975): needs local logits of the reference;
  option (d) in Phase 4.
- AgentProv (arXiv 2609.00052): tool-call fingerprints; out of scope.
- Cai et al. (arXiv 2504.04715): threat model for model substitution; TEEs as a fix.
- Harm side: Unsloth Dynamic 2.0 KL/flip discussion; "Illusion of Equivalency" (arXiv 2607.08734);
  social bias changes in quantized LMs (arXiv 2602.06181). Motivates measuring flips and KL, not
  just accuracy (Phase 5).

---

## 5. Consequences for our design

1. **Primary probe:** IRIS `short_bits_16` (the only probe with a published, control-validated
   quantization tell at 0.6B). Secondary: `short_digits_8`, `single_number_100` (expected null
   after length ablation), and the One Token battery.
2. **Sampler:** temperature 1.0, top_p 1.0, **top_k 0, min_p 0, repeat_penalty 1.0**, fixed
   seeds. This is more permissive than IRIS's unrecorded Ollama default and than llama-server's
   defaults (top_k 40, top_p 0.95, min_p 0.05). Document as a deviation; optionally run a top_k=20
   sensitivity check.
3. **Runtime:** one pinned llama-server build for every variant; a second independent fp16 run as
   the null in every comparison, exactly as IRIS does.
4. **Features:** report content-only (length-ablated) as primary, full as secondary.
5. **Expectation for the Phase 1 gate:** q4 vs fp16 at 0.6B on 16 bits should give q roughly
   0.3–0.45 with the 2.5th percentile above 0.05, and the null should sit near 0.03–0.05.
   Our quantization is our own Q4_K_M from our own f16, not Ollama's build, so some difference
   is expected.
