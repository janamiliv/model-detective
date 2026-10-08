# Which Model Am I Talking To? — Can You Detect Quantization from Outputs Alone?

Project plan and agent working agreement. Read this whole file before doing anything.

---

## 0. TL;DR

Build a reproducible experiment and a small interactive demo that answer:

> **How many queries does it take to tell a quantized open-weight LLM apart from its full-precision original, using only its text outputs? And does that point come before or after the quantization actually hurts the model?**

Deliverables: a public GitHub repo, a results write-up (README + blog post), a "Spot the model" game, and a short CV bullet. Target effort: roughly 3 to 4 weekends.

**Honest framing (important):** quantization detection from black-box outputs is already an active research area (see Section 2). We are NOT claiming a first. The project is a careful reproduction plus an extension. In every write-up say "to my knowledge" for the extension claims, and do not oversell.

---

## 1. Goals, non-goals, success criteria

### Goals
1. Reproduce a published fingerprinting/auditing baseline locally so our numbers have a reference point.
2. Run a controlled **bit-width sweep** (Q8 down to Q2) and measure **queries-to-detect** at a fixed false-positive rate.
3. Measure the **harm** side (accuracy, answer flips, KL divergence vs the fp16 reference) for the same variants, and plot detection against harm.
4. Test whether the **quantization method** (GGUF K-quants vs ONNX INT8 vs others) leaves its own fingerprint.
5. Ship a small, fun demo ("Spot the model") and a clear write-up.

### Non-goals
- Probing or auditing third-party commercial APIs without explicit permission and checking their terms of service. All core experiments run on models we host locally.
- Building an attack or evasion tool. We measure detectability and discuss defenses and limits. Do not write code whose purpose is to help a provider evade audits or to spoof fingerprints.
- Beating the state of the art. A clean, honest, reproducible result is the goal.

### Success criteria (definition of done)
- Repo installs and reproduces the headline figure from a clean clone with documented commands.
- Baseline reproduction recorded, including where it matches or differs from the papers.
- A detection-power curve per variant with confidence intervals, and a null control that stays near the nominal false-positive rate.
- A detection-vs-harm plot.
- README with problem, method, results table, limitations, and how to reproduce.
- Demo runnable locally and deployed (or recorded as a GIF if deployment is skipped).

---

## 2. Background (what the research says)

Everything below came from a web research pass in Oct 2026. Several items are preprints. **Re-verify numbers before citing them.** Where the agent only saw an abstract, that is marked.

| Work | What it does | Why it matters for us |
|---|---|---|
| Model Equality Testing (Gao, Liang, Guestrin; ICLR 2025) | Frames "which model is this API serving?" as a two-sample test using Maximum Mean Discrepancy with a Hamming string kernel. Reports median 77.4% power with ~10 samples per prompt against distortions including quantization; found 11 of 31 commercial Llama endpoints differing from reference weights. Public pip package `model-equality-testing` and a dataset of ~1M completions (includes fp32 and int8). | Baseline method and data. |
| Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test (Zhu et al.; ICLR 2026) | RUT: needs a locally deployed authentic reference model, so open-weight models only. Evaluates quantization, fine-tuning, jailbreaks, full substitution. | Alternative method. Needs logits from a local reference, which we have. |
| One Token Is Enough (Bruckner, arXiv 2607.10252, Jul 2026, preprint) | Fingerprint = distribution of answers to trivial one-word prompts (e.g. "name a random number between 1 and 100") across four languages, 1 output token per query. 165 models via OpenRouter; family recovered with 59.5% leave-one-out accuracy (chance 18.4%); 7.3% equal error rate with full battery, under 11% with 8 probe cells, ~100 queries per audit. Protocol, prompts, data, code released. **Only the abstract was read by the research pass.** | Main probe battery and primary method. |
| IRIS (Zhang et al., arXiv 2607.20860, Jul 2026, preprint) | Text-only audit using random-generation probes (random bit strings, digits) plus a classifier on visible-string features. Full text was read. Key quantization result: q4 vs fp16 on Qwen3 detectable only on a high-entropy probe; detection rate grew with size (per-query tell rate about 0.35 at 0.6B, 0.74 at 1.7B, 1.00 at 4B); a low-entropy probe's apparent signal was a length artifact removed by ablation; fp16-vs-fp16 control stayed below the floor. Code: github.com/Photen/IRIS-audit | Closest prior work. Our sweep extends its q4-vs-fp16 result. |
| AgentProv (arXiv 2609.00052) | Fingerprints through tool-call distributions rather than text. | Out of scope; mention in related work. |
| Cai et al. 2025 (arXiv 2504.04715) | Formalizes model substitution (quantization, randomized substitution, benchmark evasion); proposes TEEs as a robust fix. | Framing and threat model. |
| Harm-side work: Unsloth "Accuracy is Not All You Need" discussion; "Illusion of Equivalency" (arXiv 2607.08734); "Investigating Social Bias Changes in Quantized LMs" (arXiv 2602.06181) | Aggregate accuracy can stay flat while individual answers flip; KL divergence correlates with flips; one study reports 4-bit showing 4 to 6 times more behavioral changes than 8-bit. | Justifies measuring flips and KL, not just accuracy. |

### Known pitfalls from the literature (design around these)
1. **Greedy decoding kills the signal.** At temperature 0 repeated draws add no evidence. Sample at about temperature 1.
2. **Length and format confounds.** Always run a length/format-ablated analysis. A signal that disappears under ablation is an artifact.
3. **Probe entropy matters.** Low-entropy probes (e.g. one random bit, or a single number 1 to 100) can miss quantization entirely. Use high-entropy probes (16-bit strings, 8-digit strings).
4. **Runtime confound.** The same model on different serving stacks can be distinguishable. Keep one pinned llama.cpp build, one sampler config, one hardware setup for all variants in a comparison.
5. **Hardware non-determinism.** Even temperature 0 is not perfectly deterministic on some backends. Record hardware and build.
6. **False-positive tails.** Normal-approximation tests can be anti-conservative when the null has few tells. Use permutation or exact tests, calibrate on disjoint data, report the whole FPR distribution not just the mean. Certifying small FPRs needs thousands of reference samples.
7. **Same-family models are hard to separate.** Not our main issue (we compare variants of one model) but matters for any cross-model classifier.
8. **Spoofing and adaptive providers.** A weaker model can be tuned to fool fingerprint audits. State that our method assumes a non-adaptive provider.

---

## 3. Hypotheses (write these into `docs/prereg.md` BEFORE running the full sweep)

- **H1 (bit-width curve):** Detectability rises monotonically as bit-width falls. Q8_0 is not detectable within about 100 queries; Q4_K_M and below are, on a high-entropy probe.
- **H2 (detection-harm gap):** Quantization becomes statistically detectable at a lower compression level than the one where benchmark accuracy drops measurably. Alternative outcome (equally publishable): harm precedes detectability. Report whichever happens.
- **H3 (method fingerprint):** Different quantization methods (GGUF K-quants, ONNX INT8, optionally NF4/AWQ) leave distinguishable signatures even at matched nominal bit-width.
- **H4 (scale):** Detectability increases with model size (IRIS saw this up to 4B; we test 8B if compute allows).
- **H5 (human vs classifier, for the demo):** Humans guessing from short outputs stay near chance while the classifier does not.

Pre-registration rule: record hypotheses, metrics, exclusion rules, the number of queries per cell, and the statistical tests in `docs/prereg.md` and commit it before the sweep. Any later change goes in a dated "deviations" section.

---

## 4. Compute and hardware strategy

The user has a laptop with an older GPU (exact model and VRAM unknown) and is open to cheap cloud compute if really needed.

**First step: measure, don't guess.** Phase 0 records `nvidia-smi` output, VRAM, RAM, CPU, and OS in `docs/hardware.md`.

Rough memory estimates for weights only (verify; KV cache and runtime overhead are extra):
- 0.6B at fp16: about 1.2 GB. 1.7B: about 3.4 GB. 4B: about 8 GB. 8B: about 16 GB.
- A Q4 variant needs roughly a quarter of that (Q4_K_M is about 4.9 bits per weight, so slightly more than a quarter).

Plan:
- **Local (default):** Qwen3 0.6B and 1.7B across all variants. 4B if VRAM allows, otherwise CPU or partial GPU offload (slow but workable for one-token queries) or cloud.
- **Cloud (only if needed):** 4B fp16 reference and 8B runs, and any large re-runs. One cheap single-GPU instance for a few hours should suffice. **The agent must ask the user before spending any money**, state the expected cost, and set a hard budget cap. Never put credentials in the repo.
- One-token and short-string queries are cheap. The expensive parts are building the reference pools and the benchmark/KL evaluation, so batch and cache aggressively.

---

## 5. Repository layout

```
model-detective/
├── README.md
├── PLAN.md                  # this file
├── DECISIONS.md             # running log of decisions and deviations (agent maintains)
├── pyproject.toml           # pinned deps (uv or poetry)
├── Makefile                 # make setup / quantize / collect / analyze / figures / test
├── .github/workflows/ci.yml # lint + unit tests (no GPU)
├── configs/
│   ├── models.yaml          # model ids, sizes, licenses, revisions (pin HF commit hashes)
│   ├── variants.yaml        # quantization levels and methods
│   ├── probes.yaml          # probe prompts, languages, expected alphabets
│   └── sampler.yaml         # temperature, top_k, top_p, seed policy
├── docs/
│   ├── prereg.md
│   ├── hardware.md
│   ├── related_work.md
│   └── threat_model_and_ethics.md
├── src/modeldetective/
│   ├── serve/               # llama.cpp server wrapper, ONNX runner, request client
│   ├── quantize/            # HF -> GGUF -> llama-quantize pipeline, manifest writer
│   ├── probes/              # probe definitions, response parsers
│   ├── collect/             # sampling jobs, resumable, writes parquet
│   ├── detect/              # JS divergence, MMD (Hamming), feature+classifier, permutation tests
│   ├── harm/                # benchmark eval, flip rate, KL divergence
│   ├── analysis/            # power curves, bootstraps, plots
│   └── game/                # data prep for the demo
├── app/                     # Streamlit or Gradio demo (reads precomputed samples)
├── tests/                   # unit tests with synthetic distributions
├── data/                    # gitignored; raw samples, manifests (small summaries committed)
└── results/                 # committed final tables and figures
```

Do not commit model weights, GGUF files, or large raw data. Use `.gitignore`, and publish processed samples as a Hugging Face dataset or release asset if useful.

---

## 6. Phases and steps

Each phase has a goal, concrete steps, outputs, and a **gate** (conditions to pass before moving on). Update `DECISIONS.md` at the end of every phase.

### Phase 0 — Environment and project skeleton (about 2 to 3 hours)
**Goal:** a clean, reproducible, testable base.
1. Create the repo structure above. Initialize git, `.gitignore` (models, data, caches, `.env`).
2. Set up Python env (3.11 or newer) with a lockfile. Add ruff, pytest, pre-commit.
3. Record hardware in `docs/hardware.md` (`nvidia-smi`, RAM, CPU, OS).
4. Build llama.cpp from source at a **pinned commit** (CUDA if the GPU supports the required compute capability, otherwise CPU). Record the commit hash and build flags in `docs/hardware.md`.
5. Install Ollama only if needed for convenience; prefer `llama-server` for pinned, reproducible runs.
6. Add CI that runs lint and unit tests (no GPU, no models).
7. Add a `docs/threat_model_and_ethics.md` stub: scope is local models and consenting setups; no unauthorized API probing; no evasion tooling.
**Outputs:** green CI, documented hardware, working llama.cpp build.
**Gate:** `make test` passes; llama.cpp can load and generate from a tiny test GGUF.

### Phase 1 — Literature pin and baseline reproduction (about 1 weekend)
**Goal:** a trustworthy reference point before our own sweep.
1. Read the primary sources in Section 2 (full text of IRIS and the One Token paper; MET paper). Fill `docs/related_work.md` with one paragraph per paper plus the exact setup details (probes, temperatures, models, test statistics, sample counts).
2. Clone and run the public code: `model-equality-testing` package and the One Token fingerprint tooling (and optionally IRIS-audit). Record versions and commits.
3. Reproduce at least one published result on small models locally, e.g. the IRIS q4-vs-fp16 detection on Qwen3 0.6B using a high-entropy probe. Record where our numbers match or differ.
4. Check the licenses of every model and dataset you plan to use (Qwen and Gemma terms differ). Note them in `configs/models.yaml`.
5. Write `docs/prereg.md` (hypotheses, metrics, tests, queries per cell, exclusion rules).
**Outputs:** `related_work.md`, baseline reproduction notes with numbers, `prereg.md`.
**Gate (important):** if q4-vs-fp16 at 0.6B is not detectable on the high-entropy probe within a few hundred queries, STOP and investigate (sampler settings, probe parsing, runtime mismatch) before building more. Report to the user.

### Phase 2 — Models and quantization pipeline (about 1 weekend, can overlap with Phase 3)
**Goal:** a reproducible set of variants with a manifest.
1. Pick models (start with Qwen3 0.6B and 1.7B; add 4B; 8B on cloud later). Pin Hugging Face revisions.
2. Convert each HF checkpoint to an **f16 GGUF** (this is the reference), then produce variants with `llama-quantize`: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K.
3. Write a **manifest** (JSON/YAML) per variant: source revision, llama.cpp commit, command used, file size, SHA256, bits per weight.
4. ONNX track (feasibility first): check whether text generation with the chosen model works through ONNX Runtime (for example via onnxruntime-genai), then produce an INT8 variant. If the toolchain does not support the model cleanly, document that, and substitute bitsandbytes NF4 or AWQ only if the older GPU supports it. Do not sink more than half a day into this; H3 is the stretch hypothesis.
5. Smoke tests: each variant loads, generates coherent text, and matches the reference on a tiny sanity prompt set.
**Outputs:** variant files (local only), manifests, a size-vs-bits table.
**Gate:** every planned variant loads and generates; manifests complete.

### Phase 3 — Probe battery and data collection (about 1 weekend)
**Goal:** large, clean sample pools per (model, variant, probe).
1. Implement the probe battery in `configs/probes.yaml`:
   - **One-token probes** from the One Token paper (trivial one-word prompts such as "name a random number between 1 and 100", in several languages). Reuse the released prompts where possible.
   - **High-entropy string probes** from IRIS style: 16 random bits, 8 random digits, and 10-way digit variants.
   - **Natural-prompt set** (about 25 short prompts) for the MET-style sequence test.
   - Optional: a few Serbian prompts (Latin and Cyrillic scripts) as an extension, clearly labeled optional.
2. Write response parsers that extract the symbol or string, flag format violations, and never silently drop data (count and report them).
3. Fix sampler config: temperature about 1.0, fixed top_k/top_p (document them), no repetition penalty, one pinned llama.cpp build, same hardware. Seed policy: different seeds per run, recorded.
4. Build a resumable collection job writing parquet with columns: model, variant, probe, prompt_id, run_id, seed, response_raw, response_parsed, n_tokens, latency, build_hash.
5. Collect pools. Suggested targets: 2,000 or more samples per (variant, high-entropy probe), and 1,000 or more per one-token cell, plus **two independent fp16 runs** (reference pool and a null pool).
6. Also capture, locally, the **true next-token distribution** (logprobs) for one-token probes from each variant. This gives an "oracle" distance (KL or JS vs fp16) to compare black-box detectability against.
**Outputs:** parquet datasets, collection logs, parse-failure report.
**Gate:** null pool (fp16 vs fp16) looks statistically indistinguishable in a quick check; parse failure rate documented and low.

### Phase 4 — Detection methods and statistics (about 1 weekend)
**Goal:** implemented, tested detectors and a rigorous power analysis.
1. Implement detectors in `src/modeldetective/detect/`:
   - (a) Jensen-Shannon divergence between one-token answer distributions (One Token style).
   - (b) MMD with a Hamming string kernel (MET style); use the pip package for cross-checking.
   - (c) Visible-string feature vector plus a classifier (IRIS style; random forest is fine). Keep a length/format-invariant "content-only" feature subset.
   - (d) Optional: RUT using local logits.
2. Unit-test every statistic on **synthetic distributions with known differences** (known power, known null behavior) before touching model data.
3. Power analysis procedure: for each budget m in {1, 2, 4, 8, 16, 32, 64, 128, 256}, form episodes of m queries from the variant pool and from the reference pool; calibrate the decision threshold on a disjoint null split at alpha = 0.05; measure power on a held-out split. Use permutation or exact tests; avoid unchecked normal approximations. Bootstrap confidence intervals.
4. Primary metric: **queries-to-detect** (smallest m reaching 80% and 95% power at alpha = 0.05). Secondary: AUROC at fixed m, equal error rate.
5. Controls on every plot: fp16-vs-fp16 null (FPR must stay near alpha), and length-ablated features.
6. Report per-probe results; do not cherry-pick the best probe after the fact (the prereg decides the primary probe).
**Outputs:** detector library with tests, power curves, queries-to-detect table.
**Gate:** null FPR within tolerance; length-ablated result reported; results consistent with Phase 1 baseline or discrepancy explained.

### Phase 5 — Harm measurement and the detection-harm gap (about 1 weekend, can overlap with Phase 4)
**Goal:** quantify how much each variant actually changes behavior.
1. Choose a benchmark subset (for example a few hundred to a thousand multiple-choice questions from MMLU or a similar set; fix the subset and seed). Use identical prompts and decoding for all variants (greedy is fine here since we want task accuracy, not distribution).
2. Metrics per variant vs the fp16 reference: accuracy, **flip rate** (correct to wrong and wrong to correct), and **KL divergence** of next-token distributions on a held-out text sample (llama.cpp has KL-divergence tooling; check the current flag names in the pinned build).
3. Build the **detection-harm plot**: x = harm (accuracy drop, flip rate, or KL), y = queries-to-detect, one point per variant per model size. Mark where detectability appears relative to where harm becomes significant.
4. State H2's outcome plainly, whichever direction it goes.
**Outputs:** harm table, detection-vs-harm figure.
**Gate:** harm metrics reproducible; plot reviewed for honesty (axes, CIs, no truncated scales).

### Phase 6 — Method fingerprinting, H3 (stretch; about half a weekend)
**Goal:** can we identify the quantization method, not just detect quantization?
1. For matched nominal bit-width (e.g. around 8-bit and around 4-bit), compare GGUF vs ONNX INT8 (and NF4/AWQ if available).
2. Train a multi-class classifier on high-entropy probe responses; report a confusion matrix, accuracy vs query budget, and a null where labels are shuffled.
3. Beware of confounds: different runtimes. If the method difference is confounded with the runtime, say so explicitly (as IRIS cautions about provider-versus-precision confounds).
**Gate:** skip cleanly and document if Phase 2's ONNX track was infeasible. Do not block the project on this.

### Phase 7 — Scale-up, H4 (optional; needs cloud)
**Goal:** extend to 4B and 8B.
1. Ask the user for approval and a budget cap before starting any cloud instance.
2. Reuse the same pipeline; run only the variants needed to test the size trend (e.g. fp16, Q8_0, Q4_K_M, Q2_K).
3. Tear down instances immediately after; record cost and time in `DECISIONS.md`.
**Outputs:** extra rows in the results table; updated figures.

### Phase 8 — "Spot the model" demo (about 1 weekend)
**Goal:** a fun, shareable interface.
1. Precompute a pool of anonymized responses per variant (no live inference needed, so hosting is cheap).
2. Build a Streamlit or Gradio app: shows a handful of responses (one-token answers or short strings) from a hidden variant; the player guesses fp16 vs quantized (or which bit-width); show the classifier's guess and confidence side by side; show the detection-vs-queries curve.
3. Do not collect personal data. If guesses are stored for the H5 comparison, store only anonymous aggregates and say so in the app.
4. Deploy (for example Hugging Face Spaces) or, at minimum, record a short GIF demo for the README.
**Outputs:** `app/`, deployed link or GIF.

### Phase 9 — Write-up, polish, and release (about 1 weekend)
1. README structure: one-paragraph problem, headline result and figure, how it works (diagram), results table, limitations, reproduce commands, links.
2. Blog post (800 to 1,500 words): story, method, honest results including negative ones, what surprised us, what is not claimed.
3. Limitations section must cover: small models only, one runtime, controlled local setup (not commercial APIs), non-adaptive provider assumption, preprint sources, single hardware.
4. Cross-check every number against the sources and against our own result files. Add a `results/README.md` mapping each figure to the script that generated it.
5. Pin the repo on GitHub, add a profile README linking it, and add topics.
6. Draft two CV bullets with concrete numbers, for example: "Measured how many black-box queries are needed to detect INT8 to 2-bit quantization across N model sizes; found detection precedes accuracy loss at X bits" (fill only with real results).
7. Optional: post to relevant communities (including the Serbian AI Society) once the repo is clean.

---

## 7. Suggested timeline

| Week | Focus |
|---|---|
| 1 | Phases 0 and 1 (setup, reading, baseline reproduction, prereg) |
| 2 | Phases 2 and 3 (variants, probes, data collection) |
| 3 | Phases 4 and 5 (detectors, power curves, harm metrics, main plot) |
| 4 | Phases 8 and 9, plus Phase 6 or 7 if time and budget allow |

Cut order if time runs short: Phase 7 first, then Phase 6, then reduce Phase 8 to a GIF.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Baseline does not reproduce | Phase 1 gate; check sampler, parsing, runtime; ask the user before proceeding |
| Older GPU cannot run needed builds | CPU or offload fallback; small models first; cloud only with approval |
| ONNX generation toolchain not viable | Time-box to half a day; fall back to GGUF-only and drop H3 |
| Result is "not detectable" for mild quantization | That is a valid finding; report the detection floor honestly |
| Confounds (runtime, length, hardware) | Pin builds, ablate length, null controls, document |
| Overclaiming novelty | Use "to my knowledge", cite prior work prominently |
| Scope creep | Follow the cut order; log extras in `DECISIONS.md` instead of building them |

---

## 9. Working agreement for the coding agent

1. Read this file first. Work phase by phase. Do not skip gates.
2. Make small commits with clear messages. Run `make test` before committing.
3. Never commit weights, GGUF files, large data, secrets, or `.env`.
4. Pin versions: llama.cpp commit, Hugging Face revisions, Python deps. Record them.
5. Write unit tests for statistical code using synthetic data with known answers.
6. Commit `docs/prereg.md` before running the full sweep. Log deviations with dates.
7. **Ask the user before:** spending cloud money, changing hypotheses or primary metrics, dropping a phase, or deviating from the scope in Section 1.
8. Report parse failures, null-control failures, and surprising results immediately rather than burying them.
9. Do not probe third-party commercial APIs. Do not build evasion or spoofing tools.
10. Keep `DECISIONS.md` current: what was decided, why, and what alternatives were rejected.
11. When uncertain about a number from a paper, re-check the paper; do not rely on this plan's summary.

---

## 10. Open questions to resolve early

1. Exact GPU model and VRAM (decides how much runs locally).
2. Which Qwen3 and Gemma sizes and licenses are acceptable to publish results for.
3. Whether ONNX Runtime text generation works for the chosen model on this machine.
4. Current flag names for llama.cpp's KL-divergence and perplexity tools in the pinned build.
5. Whether to include Serbian (Latin and Cyrillic) probes in the main analysis or as a labeled extension (default: extension).
6. Whether the user wants the demo deployed publicly or only as a recorded GIF.

---

## 11. Sources

- Model Equality Testing: https://www.iclr.cc/virtual/2025/poster/29704 and https://pypi.org/project/model-equality-testing/
- Rank-Based Uniformity Test: https://arxiv.org/abs/2506.06975v2
- One Token Is Enough: https://arxiv.org/abs/2607.10252
- IRIS: https://arxiv.org/html/2607.20860v1 (code: https://github.com/Photen/IRIS-audit)
- AgentProv: https://arxiv.org/html/2609.00052v1
- Are You Getting What You Pay For?: https://arxiv.org/abs/2504.04715
- Reference fingerprint tool for the One Token paper: https://github.com/ToseaAI/llm-fingerprint-detector
- Unsloth Dynamic 2.0 GGUFs (flips and KL divergence): https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
- The Illusion of Equivalency: https://arxiv.org/pdf/2607.08734
- Social bias changes in quantized LMs: https://arxiv.org/pdf/2602.06181
- GGUF quantization workflow example: https://dev.to/ginigen_ai_010d9180cbbdb1/4-bit-gguf-quality-for-moe-models-why-only-3b-of-180b-params-fire-and-how-to-prove-parity-16a9
- Running Gemma with Ollama / llama.cpp: https://ai.google.dev/gemma/docs/integrations/ollama?hl=en
