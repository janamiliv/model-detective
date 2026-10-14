# Hardware and runtime

Recorded 2026-10-08/09. All local experiments run on this machine unless DECISIONS.md says otherwise.

## Machine

| Item | Value |
|---|---|
| Model | Lenovo 82B5 (laptop) |
| OS | Windows 11 Education, build 26200 |
| CPU | AMD Ryzen 7 4800H, 8 cores / 16 threads, 2.9 GHz base |
| RAM | 15.4 GB usable, DDR4-3200 |
| GPU | NVIDIA GeForce GTX 1650 (laptop), 4096 MiB VRAM, compute capability 7.5 (Turing) |
| NVIDIA driver | 535.98 (CUDA 12.2) until 2026-10-14; **617.42 (CUDA 13.4) from 2026-10-14** |
| Disk | D: 182 GB, 81 GB free at start |

Note: desktop apps (chat clients, browsers) hold about 1.7 GB of VRAM when open. Close them before collection runs.

## What fits in 4 GB VRAM

Weights only; KV cache and runtime overhead are extra.

| Model | f16 | Q8_0 | Q4_K_M | Fully on GPU? |
|---|---|---|---|---|
| Qwen3 0.6B | ~1.2 GB | ~0.65 GB | ~0.4 GB | yes, all variants |
| Qwen3 1.7B | ~3.4 GB | ~1.8 GB | ~1.1 GB | f16 is tight; expect partial offload |
| Qwen3 4B | ~8 GB | ~4.3 GB | ~2.5 GB | no for f16/Q8; partial offload or CPU |
| Qwen3 8B | ~16 GB | ~8.7 GB | ~4.9 GB | no; cloud only |

## Software

| Tool | Version |
|---|---|
| Python | 3.11.9 |
| uv | 0.12.24 |
| git | 2.53.0.windows.1 |
| GNU make | 4.4.1 (ezwinports, user scope) |
| cmake / nvcc / MSVC | not installed (no source builds) |

## llama.cpp (pinned)

| Item | Value |
|---|---|
| Release tag | `v0.6.0` (stable), which points to nightly build `b11429` |
| Commit | `d81235049384534c167caea52b85a694f6103d14` |
| Binary | `llama-b11429-bin-win-cuda-12.4-x64.zip`, SHA256 `dd6df685c1024e6aa55ca55ad35038692d5e284d4dc895e7a646c3245d3d14ff` |
| CUDA runtime | `cudart-llama-bin-win-cuda-12.4-x64.zip`, SHA256 `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6` |
| Reported version | `0.6.0-dev (build 11429, commit d81235049)`, built with Clang 20.1.8 for Windows x86_64 |
| Install path | `tools/llama.cpp/bin/` (gitignored) |
| Source | https://github.com/ggml-org/llama.cpp/releases/tag/b11429 |

No source build: cmake, the CUDA toolkit and MSVC are not installed. The prebuilt binary is pinned by build number and SHA256 instead of a local build flag record.

### Gate test (Phase 0)

Model: `Qwen/Qwen3-0.6B-GGUF`, file `Qwen3-0.6B-Q8_0.gguf`, HF revision `23749fefcc72300e3a2ad315e1317431b06b590a`, SHA256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`, stored at `models/qwen3-0.6b/` (gitignored).

| Backend | Result |
|---|---|
| CUDA (`-ngl 99`) | **Fails**: `CUDA error: invalid device function`. The release build compiles compute capability 7.5 as PTX only (`75-virtual` in `ggml/src/ggml-cuda/CMakeLists.txt`), and PTX from CUDA 12.4 cannot be JIT-compiled by driver 535.98, which supports CUDA 12.2 at most. |
| Vulkan (`llama-b11429-bin-win-vulkan-x64.zip`, SHA256 `1bfe78ad9168b79fa02bf67f6af9f5e17a966d824d77238517f7bef12ac73b36`) | **Hangs** at startup with no log output for over 10 minutes; killed. Not investigated further. |
| CPU (`--device none -ngl 0`) | **Passes.** Greedy completion of "The capital of Serbia is" gives "Belgrade, and the capital of Montenegro is Podgorica. ..." |

**Fix applied 2026-10-14:** driver updated to 617.42 (CUDA 13.4). The CUDA gate now passes with the same pinned b11429 CUDA-12.4 binary: identical greedy output to the CPU run ("Belgrade, and the capital of Montenegro is Podgorica..."), 82 tokens/s single-stream vs 13 on CPU. All 29/29 layers offload (`load_tensors: offloaded 29/29 layers to GPU`). The first request after each process start pays a one-time PTX JIT compile of 13-21 s per quant type (the driver caches it afterwards).

### Throughput (Qwen3 0.6B, llama-server, 8 slots, temperature 1, 16-bit probe)

| Backend | Variant | Steady state | Notes |
|---|---|---|---|
| CPU (8 threads) | f16 | ~250 samples/min, ~86 gen tok/s | measured with 24-token cap, mean 20 tokens |
| CUDA | f16 | 183 samples/min, 84 gen tok/s | GPU ran at 855 MHz / 22 W under load (max 1785 MHz / 50 W) |
| CUDA | Q4_K_M | 244 samples/min, 138 gen tok/s | |

The GPU is not faster than the CPU at 0.6B under the conditions measured: Windows "Balanced" power plan, and other apps (a video player decoding at ~33% SM, an animated wallpaper) sharing the GPU. No thermal or power-cap throttling was flagged. For collection runs: close GPU-using apps and consider the High performance power plan. The GPU matters more at 1.7B and 4B, where CPU decode slows down faster than GPU decode.

**Backend policy (prereg §4):** every comparison uses one backend. Phase 1 cells were collected on CPU and re-collected on CUDA as a cross-backend check (see `results/phase1_iris_repro*.json`).

### Tools present in the build

`llama-cli`, `llama-completion`, `llama-server`, `llama-quantize`, `llama-perplexity`, `llama-imatrix`, `llama-bench`, `llama-tokenize`, `llama-gguf-split`.

KL-divergence flags in this build (answers PLAN.md open question 4): `llama-perplexity --save-all-logits FILE` on the reference, then `llama-perplexity --kl-divergence --kl-divergence-base FILE` on the variant. Quantization types available: `Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K` (plus `F16` for the reference).
