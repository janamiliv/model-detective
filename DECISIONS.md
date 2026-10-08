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
