"""KL divergence of next-token distributions, variant vs f16, via llama-perplexity (b11429).

Two-step flow (flag names verified in docs/hardware.md):
  1. reference:  llama-perplexity -m f16.gguf -f text.txt --save-all-logits ref.logits
  2. variant:    llama-perplexity -m var.gguf -f text.txt --kl-divergence-base ref.logits --kl-divergence
The variant run prints mean KL, its standard error, and top-1 agreement; we parse those lines.
The held-out text is configs/kl_text.txt (fixed; must not overlap the probes).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

LLAMA_BIN = Path("tools/llama.cpp/bin")


def save_reference_logits(
    model: Path, text: Path, out: Path, backend: str, ctx: int = 512, chunks: int = 64
) -> None:
    cmd = [
        str(LLAMA_BIN / "llama-perplexity.exe"),
        "-m",
        str(model),
        "-f",
        str(text),
        "-c",
        str(ctx),
        "--chunks",
        str(chunks),
        "--save-all-logits",
        str(out),
    ]
    cmd += ["-ngl", "99"] if backend == "cuda" else ["--device", "none", "-ngl", "0"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def kl_against_reference(
    model: Path, text: Path, ref_logits: Path, backend: str, ctx: int = 512, chunks: int = 64
) -> dict:
    cmd = [
        str(LLAMA_BIN / "llama-perplexity.exe"),
        "-m",
        str(model),
        "-f",
        str(text),
        "-c",
        str(ctx),
        "--chunks",
        str(chunks),
        "--kl-divergence-base",
        str(ref_logits),
        "--kl-divergence",
    ]
    cmd += ["-ngl", "99"] if backend == "cuda" else ["--device", "none", "-ngl", "0"]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    log = res.stdout + res.stderr
    return parse_kl_output(log)


def parse_kl_output(log: str) -> dict:
    """Pull the summary numbers out of llama-perplexity's KL report; raise if absent."""
    out: dict = {}
    for key, pat in {
        "mean_kl": r"Mean\s+KLD:\s*([0-9.]+)\s*±\s*([0-9.]+)",
        "max_kl": r"Maximum\s+KLD:\s*([0-9.]+)",
        "kl_99pct": r"99\.9%\s+KLD:\s*([0-9.]+)|99\.0%\s+KLD:\s*([0-9.]+)",
        "median_kl": r"Median\s+KLD:\s*([0-9.]+)",
        "top1_agreement": r"Same top p:\s*([0-9.]+)\s*±\s*([0-9.]+)\s*%|Same top:\s*([0-9.]+)",
        "mean_ppl_ratio": r"Mean\s+ln\(PPL\(Q\)/PPL\(base\)\):\s*([-0-9.]+)",
    }.items():
        m = re.search(pat, log)
        if m:
            vals = [g for g in m.groups() if g is not None]
            out[key] = float(vals[0])
            if key in ("mean_kl", "top1_agreement") and len(vals) > 1:
                out[key + "_se"] = float(vals[1])
    if "mean_kl" not in out:
        raise RuntimeError(
            "could not find 'Mean KLD' in llama-perplexity output; check flag names:\n"
            + log[-2000:]
        )
    return out
