# Decisions log

Running log of decisions and deviations. Newest at the bottom. Format: date, decision, why, alternatives rejected.

## 2026-10-08 — Phase 0 start

**Hardware measured (full details go in `docs/hardware.md`).** GTX 1650 with 4 GB VRAM, driver 535.98 / CUDA 12.2, Ryzen 7 4800H (8c/16t), 15.4 GB RAM, 81 GB free disk. Consequence: Qwen3 0.6B and 1.7B run fully on GPU; 4B f16 reference does not fit in VRAM and needs offload, CPU, or cloud.

**Repo initialized in the existing folder, branch `main`.** Folder name kept as is for now; the GitHub repo name (`model-detective` per PLAN.md) can differ from the local folder. Alternative rejected: renaming the folder now, which gains nothing until publication.

**Line endings forced to LF via `.gitattributes`, `core.autocrlf=false`.** Why: Makefile, shell scripts, and CI run on Linux; mixed CRLF on Windows causes spurious diffs and broken scripts.

**`.gitignore` blocks weights, GGUF, ONNX, parquet, `data/`, local builds, `.env`.** `data/README.md` is whitelisted so the folder's purpose is documented. Small summaries must be added with `git add -f` deliberately.

**Open (to decide next):** llama.cpp as pinned release binary vs. source build with CUDA (cmake, CUDA toolkit, MSVC not installed; recommendation is the release binary).
