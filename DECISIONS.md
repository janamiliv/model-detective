# Decisions log

Running log of decisions and deviations. Newest at the bottom. Format: date, decision, why, alternatives rejected.

## 2026-10-08 — Phase 0 start

**Hardware measured (full details go in `docs/hardware.md`).** GTX 1650 with 4 GB VRAM, driver 535.98 / CUDA 12.2, Ryzen 7 4800H (8c/16t), 15.4 GB RAM, 81 GB free disk. Consequence: Qwen3 0.6B and 1.7B run fully on GPU; 4B f16 reference does not fit in VRAM and needs offload, CPU, or cloud.

**Repo initialized in the existing folder, branch `main`.** Folder name kept as is for now; the GitHub repo name (`model-detective` per PLAN.md) can differ from the local folder. Alternative rejected: renaming the folder now, which gains nothing until publication.

**Line endings forced to LF via `.gitattributes`, `core.autocrlf=false`.** Why: Makefile, shell scripts, and CI run on Linux; mixed CRLF on Windows causes spurious diffs and broken scripts.

**`.gitignore` blocks weights, GGUF, ONNX, parquet, `data/`, local builds, `.env`.** `data/README.md` is whitelisted so the folder's purpose is documented. Small summaries must be added with `git add -f` deliberately.

**Open (to decide next):** llama.cpp as pinned release binary vs. source build with CUDA (cmake, CUDA toolkit, MSVC not installed; recommendation is the release binary).

## 2026-10-08 — Phase 0 step 2: Python environment

**uv (0.12.24) with `uv.lock` as the single source of truth; hatchling build backend.** Why: fast, lockfile by default, one tool for venv + deps + tool running. Alternatives rejected: poetry (slower, second config format), plain pip (no lockfile).

**Base deps kept light** (numpy, pandas, pyarrow, scipy, scikit-learn, matplotlib, pyyaml, httpx, tqdm, pydantic, typer). Heavy or phase-specific deps are optional extras: `quantize` (torch, transformers, gguf; Phase 2), `baseline` (`model-equality-testing`; Phase 1), `app` (streamlit; Phase 8). They are declared but not yet installed.

**Dev group:** ruff (lint + format, line length 100), pytest with `gpu` and `slow` markers so CI runs `-m "not gpu and not slow"`, pre-commit (whitespace, EOL, yaml/toml, large-file and private-key checks, ruff), hypothesis for property tests of the statistics code later.

**GNU make 4.4.1 from ezwinports via winget at user scope.** The GnuWin32 installer needs admin and was cancelled. The Makefile only delegates to `uv run`, so it stays trivially cross-platform. New shells need a restart to pick up `uv` and `make` on PATH.

**Gate status:** `make test` passes (2 smoke tests), `make lint` clean, pre-commit hooks installed.

## 2026-10-09 — Phase 0 steps 3, 4, 6, 7: hardware doc, llama.cpp, CI, ethics stub

**llama.cpp as a pinned prebuilt release, not a source build.** Pinned to stable tag `v0.6.0` = build `b11429` = commit `d812350`, with SHA256s of the zips in `docs/hardware.md`. Why: no cmake, CUDA toolkit or MSVC on the machine, and a build-number plus hash pins the runtime just as well. Alternative rejected: source build (half an evening of toolchain setup, and the same confound control is achieved by the hash).

**CUDA gate fails on this machine; CPU gate passes.** The release binary ships compute capability 7.5 (GTX 1650) as PTX only, and the installed driver 535.98 (CUDA 12.2) cannot JIT PTX from CUDA 12.4. Vulkan build hung at startup. **Action for the user:** update the NVIDIA driver to 550 or later, then re-run the CUDA gate. Until then, all runs are CPU. For this project's model sizes that is slow but not blocking.

**Gate GGUF is the official `Qwen/Qwen3-0.6B-GGUF` Q8_0** (revision pinned in hardware.md). It doubles as the Phase 1 baseline model. Our own f16 reference and the K-quant sweep still come from Phase 2's conversion pipeline.

**CI** runs on ubuntu-latest with uv 0.12.24 pinned, `uv sync --locked`, ruff check + format check, and `pytest -m "not gpu and not slow"`. No GPU, no models.

**Open question 4 (KL flags) resolved:** see `docs/hardware.md`, "Tools present in the build".

**Phase 0 gate:** `make test` passes; llama.cpp loads and generates from a GGUF (on CPU). Passed with the GPU caveat above.

## 2026-10-14 — Phase 1: literature pin and baseline reproduction setup

**Primary sources read in full** (IRIS, One Token, MET) plus their code; details in `docs/related_work.md`. Corrections to PLAN.md's summary: One Token uses `max_tokens` 16 (not 1) and never tests quantization; its "8 cells" EER is a mean over random subsets; the IRIS 0.35/0.74/1.00 numbers come from the across-sizes paragraph (length-ablated, 5-fold), not Table 8 (0.37 ablated / 0.42 full at 0.6B). MET's quantization is bitsandbytes int8/nf4, not GGUF.

**Reimplement IRIS's probe, parser, features and tell-rate protocol instead of running its scripts.** Why: its collectors speak only Ollama's native API; we serve with llama-server. Our implementation lives in `src/modeldetective/{probes/iris.py, detect/features.py, detect/tell_rate.py}` and is unit-tested on synthetic data (null near alpha, strong shift detected). Alternative rejected: installing Ollama (second, unpinned runtime; defeats the runtime-confound control).

**Sampler for all comparisons: temperature 1.0, top_p 1.0, top_k 0, min_p 0, repeat_penalty 1.0, fixed per-request seeds, Qwen3 thinking disabled, no system prompt for IRIS probes.** Why: IRIS never set top_k, so Ollama's default (unrecorded) applied; we choose the fully untruncated distribution because truncation itself interacts with quantization. Deviation from IRIS, recorded. llama-server's own defaults (top_k 40, top_p 0.95, min_p 0.05) are never used.

**Reference and variant are our own GGUFs** (f16 via convert_hf_to_gguf at b11429; Q4_K_M via llama-quantize at b11429), not Ollama's library builds. Hashes in `configs/models.yaml`. The official Qwen Q8_0 GGUF is kept only as a cross-check.

**Phase 1 reproduction cells:** IRIS Table 8 at 0.6B: probes `short_bits_16` and `single_number_100`; 200 responses per label; independent second f16 run (250 per probe) as null. All on CPU (8 threads, 8 server slots), about 250 samples/min at f16.

**Third-party code licenses:** IRIS code PolyForm Noncommercial 1.0.0, data CC BY-NC 4.0 (fine for this non-commercial project; cite and keep notices if redistributing). One Token author code MIT, data CC BY 4.0; verbatim prompts copied to `configs/one_token_prompts.zenodo.json`. MET code MIT; dataset license unstated. Qwen3 0.6B/1.7B/4B are Apache-2.0: results may be published.

**Phase 1 gate result:** appended below once the run finishes.

**Phase 1 gate result (2026-10-14): PASSED.** Q4_K_M vs f16 at 0.6B on 16 random bits, content-only features: q = 0.39 [0.25, 0.55], AUROC 0.83, clears the alpha floor; f16-vs-f16 null q = 0.06 [0.01, 0.15]. IRIS reported 0.37 [0.17, 0.51] and null 0.03. Full table in `results/phase1_iris_repro.md`. Full-feature q is higher than IRIS (0.56 vs 0.42); attributed to sampler/runtime/build differences and the reason content-only is primary.

**Finding to carry into Phase 3: the IRIS number parser is defeated by prompt echo at 0.6B.** The first integer in 95–99% of responses is the "1" from the prompt text. The `single_number_100` null result is therefore a parser artifact. Decision: Phase 3 number parser takes the last integer and flags echoed-only rows; documented in `docs/prereg.md` §7.

**Pre-registration drafted (`docs/prereg.md`), not frozen.** Needs owner review; freeze by tagging `prereg-v1` before collecting beyond the Phase 1 cells.

## 2026-10-14 (evening) — NVIDIA driver updated, CUDA gate passes

**Driver 535.98 → 617.42 (CUDA 13.4).** The pinned b11429 CUDA-12.4 binary now runs: 29/29 layers offload, greedy output identical to CPU, 82 tok/s single-stream. Phase 0 GPU caveat closed.

**GPU is not a throughput win at 0.6B under current conditions.** Batched sampling: CUDA f16 183/min vs CPU ~250/min; CUDA Q4_K_M 244/min. Under load the GPU sat at 855 MHz / 22 W of 1785 MHz / 50 W with no throttle flags, on the Balanced power plan, while a video player was decoding on it. Decision: keep CPU as the default backend for 0.6B collection; use CUDA for 1.7B and 4B where it should pull ahead; never mix backends within one comparison. Before long GPU runs, close GPU-using apps and consider the High performance power plan (user's call; not changed by the agent).

**Phase 1 cells re-collected on CUDA** (`BACKEND=cuda`, separate `data/phase1_iris_cuda/` and `results/phase1_iris_repro_cuda.json`) as a cross-backend check of the baseline. Result appended below.

**CUDA cross-backend result (2026-10-14):** 16-bit content-only q = 0.33 [0.17, 0.49] on CUDA vs 0.39 [0.25, 0.55] on CPU; nulls 0.06 on both. Consistent.

**Runtime confound measured and negative:** f16-CPU vs f16-CUDA with disjoint seeds is at the floor (q 0.05–0.06, AUROC 0.53). Backend is not a confound for this build; the prereg's "one backend per comparison" rule stays as a precaution.

**Seed-sharing bug found and fixed.** Same seed reproduces identical text across backends 94% of the time, and across f16/Q4_K_M 9% of the time on 16 bits. Phase 1 pools shared seeds between reference and substitute, biasing q downward (conservative; gate still passes). `seed_for` now derives disjoint ranges per (model, variant, run, probe); prereg §4 updated. Phase 1 data is kept as is and labeled; Phase 3 collection uses the new rule.

## 2026-10-15 — Phase 2: quantization pipeline

**Pipeline script** `src/modeldetective/quantize/pipeline.py`: HF → f16 (convert_hf_to_gguf at b11429) → Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K (llama-quantize b11429, no importance matrix so variants differ only in bit-width). One JSON manifest per file (revision, build, command, size, SHA256, parameter count, bits/weight, smoke output). Resumable. Summary table committed at `results/variants_table.md`; manifests and GGUFs stay local.

**Qwen3 0.6B: all seven files built and smoke-tested.** Q2_K is degenerate on the greedy smoke prompt ("20200000000..."), Q3_K_M loses the factual answer. Expect the prereg's degenerate-variant rule to apply to Q2_K at 0.6B.

**Seed determinism pre-check (prereg §4) passed on both backends:** 20 same-seed requests, serial and 4-way concurrent, CPU and CUDA, each gave exactly one distinct text. Seeds are therefore true replay handles on this build, which is also why pools must have disjoint seeds.

**Pinned revisions:** Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e; Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c (to be written into configs/models.yaml with hashes once built).

**Qwen3 1.7B: all seven files built and smoke-tested.** Coherent down to Q3_K_M; Q2_K degrades to word salad.

**Number parser for Phase 3 implemented** (`single_number_100_last`, parser `number_last`: last non-echo integer, `echo` flag). On the Phase 1 data it recovers usable in-range answers for 48% (f16), 38% (Q4_K_M) and 43% (f16 null) of responses; the remainder never state a number within the 48-token cap. That is below the prereg's 50% parse threshold at 0.6B, so the number probe remains secondary and its 0.6B cells will be flagged. The IRIS first-integer parser is kept unchanged for the baseline comparison.

**`configs/models.yaml` is now generated** from the local manifests by `quantize/sync_config.py` (hashes, bits/weight, revisions); notes stay hand-edited.

**ONNX track (H3) feasibility: GO, time-boxed to half a day.** Research result: onnxruntime-genai's model builder supports `Qwen3ForCausalLM` (added Apr 2025, PR #1428); latest 0.18.0 (2026-10-14) offers `--precision int8` (MatMulNBits 8-bit weight-only, since 0.15.0) and `int4` with `algo_config={rtn,k_quant}` and `block_size`, which is structurally comparable to GGUF Q8_0 / Q4 (block-wise weight-only). CPU execution provider first; CUDA optional (needs a CUDA toolkit for genai-cuda, GTX 1650 sm_75 is in ORT's arch list). Sampling: `do_sample=True, temperature=1.0, top_k=0, top_p=1.0, random_seed=N` (note: default `top_p=0.0` with `top_k=0` collapses to argmax). Pitfall: genai's chat-template call has no kwargs, so `enable_thinking=False` must be rendered with the HF tokenizer. Alternative rejected: Optimum export + `quantize_dynamic` (W8A8, CPU-only kernels; a poorer analogue of Q8_0). Caveat for H3: the ONNX runtime is a different serving stack, so method and runtime are confounded, exactly as IRIS warns; report it that way. Items marked UNVERIFIED in the research (exact CUDA minor of the 0.18.0 wheel, DirectML package status) do not affect the CPU route.

**Qwen3 4B: all seven files built and smoke-tested** (CPU; f16 is 7.7 GB and loads from RAM). Two smoke tests first returned empty while the ONNX builder was running concurrently; re-run alone they pass. Lesson: do not overlap RAM-heavy jobs on this 15 GB machine. Phase 2 GGUF gate: 21 files, every one loads and generates; manifests complete; table in `results/variants_table.md`.

**ONNX INT8 export done via onnxruntime-genai 0.18.0 builder, CPU EP, block size 32.** Two practical findings: (1) the 0.18.0 builder requires transformers 5 while llama.cpp's converter pins 4.57.6, so the builder runs in an isolated `uv run --no-project --with ...` environment; the genai *runtime* stays in the main venv (extra `onnx`). (2) The builder keeps embeddings and the LM head in fp32: the INT8 export is 1,253 MiB for 752M parameters, about 13.3 bits/weight overall versus 8.5 for GGUF Q8_0, which quantizes those matrices too. For H3 the comparison is therefore "ONNX builder INT8/INT4 recipe" vs "GGUF K-quant recipe", not a matched bit-width, and the runtime also differs. Report H3 with both confounds stated.

**ONNX INT8 smoke test passed** through `serve/onnx_runner.py`: temperature-1 multinomial sampling with `do_sample=True, top_k=0, top_p=1.0`, HF-rendered chat template with thinking off, seeds reproduce exactly, 16-bit probe parses 10/10, ~16 tok/s single-stream on CPU. INT4 (RTN, block 32) export in progress.

**ONNX INT4 (RTN, block 32) export done:** 375 MiB, about 4.2 bits/weight overall, so this mode quantizes the embeddings too (inferred from size). Quality is harsher than GGUF Q4_K_M: garbled factual answer, 16-bit probe parses 4/10 because it inserts spaces. Manifests in `models/qwen3-0.6b/onnx/manifests/`, summary in `results/onnx_variants.md`.

**Phase 2 gate: PASSED (2026-10-15).** 21 GGUF variants across three sizes plus two ONNX variants at 0.6B, all load and generate, manifests complete, `configs/models.yaml` generated from manifests. Time spent on the ONNX track: about two hours, within the half-day box. Open for Phase 3: the pre-registration freeze (owner's confirmation), then collection.

## 2026-10-15 — Storage: model files moved to the external drive

**`models/` (54.2 GB, 135 files) moved to `E:\model-detective\models` (WD My Passport USB HDD, 1.8 TB) with robocopy; an NTFS junction at `models/` keeps every path in configs and code unchanged.** D: went from 24 GB to 78 GB free. `tools/` (llama.cpp binaries) and `.venv` stay on the NVMe for start-up speed. Cost: model loads read from USB (about 65–75 MB/s), so a 4B f16 server start takes roughly two minutes instead of seconds; that happens once per pool. Integrity: every manifest hash re-verified after the move with `quantize/verify.py` (result below). Lesson from the move: robocopy preallocates destination files, so hashing or sizing a file mid-copy is meaningless; verify only after the job ends.
