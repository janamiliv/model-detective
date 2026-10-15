"""Greedy MMLU evaluation of one GGUF through llama-server; writes one parquet per (model, variant, run).

Decoding is greedy (temperature 0) because we want task behaviour, not a distribution. The answer
is the first non-space character of the completion, uppercased, if it is one of A-D; otherwise
the response is counted as invalid (and wrong). Logprobs of the four letters at the first
position are also stored for later analysis.

    uv run python -m modeldetective.harm.eval --models qwen3-0.6b --backend cpu [--variants f16 Q4_K_M]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml

from modeldetective.collect.sweep import MODELS_CFG, plan_pools
from modeldetective.harm.mmlu import LETTERS, format_prompt, load_subset
from modeldetective.serve import LlamaClient, LlamaServer, SamplerConfig

OUT_ROOT = Path("data/harm")
GREEDY = SamplerConfig(
    temperature=0.0, top_p=1.0, top_k=1, min_p=0.0, repeat_penalty=1.0, max_tokens=4
)


def parse_letter(text: str) -> str | None:
    t = text.strip().lstrip("(").upper()
    return t[0] if t and t[0] in LETTERS else None


def eval_pool(
    model_path: Path,
    label: str,
    questions: list[dict],
    out: Path,
    backend: str,
    parallel: int,
    threads: int,
) -> Path:
    if out.exists():
        return out
    gpu_kw = (
        {"n_gpu_layers": 0, "device": "none"}
        if backend == "cpu"
        else {"n_gpu_layers": 99, "device": None}
    )
    srv = LlamaServer(
        model_path=model_path,
        port=int(os.environ.get("MD_PORT", "8103")),
        parallel=parallel,
        ctx_size=1024 * parallel,
        threads=threads,
        startup_timeout_s=600,
        log_path=out.with_suffix(".server.log"),
        **gpu_kw,
    )
    with srv:
        build = srv.build_info()
        client = LlamaClient(srv.base_url)

        def one(q: dict) -> dict:
            t0 = time.perf_counter()
            r = client.complete(format_prompt(q), GREEDY, seed=0, n_logprobs=10)
            letter = parse_letter(r.text)
            top = {}
            if r.logprobs:
                for cand in r.logprobs[0].get("top_logprobs", []):
                    tok = cand["token"].strip().upper()
                    if tok in LETTERS and tok not in top:
                        top[tok] = cand["logprob"]
            return {
                "label": label,
                "qid": q["qid"],
                "subject": q["subject"],
                "gold": q["answer"],
                "response_raw": r.text,
                "pred": letter,
                "correct": letter == q["answer"],
                "invalid": letter is None,
                "letter_logprobs": json.dumps(top),
                "latency": time.perf_counter() - t0,
                "build_hash": build,
            }

        with cf.ThreadPoolExecutor(parallel) as ex:
            rows = list(ex.map(one, questions))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--variants", nargs="*", default=None, help="subset, e.g. f16 Q4_K_M")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    models_cfg = yaml.safe_load(MODELS_CFG.read_text(encoding="utf-8"))
    questions = load_subset()
    out_dir = OUT_ROOT / a.backend
    for key in a.models:
        for pool in plan_pools(key, models_cfg):
            if a.variants and pool.variant not in a.variants:
                continue
            label = f"{pool.model}__{pool.variant}__run{pool.run_id}"
            t0 = time.perf_counter()
            path = eval_pool(
                pool.file,
                label,
                questions,
                out_dir / f"{label}.parquet",
                a.backend,
                a.parallel,
                a.threads,
            )
            df = pd.read_parquet(path)
            print(
                f"{label}: acc={df['correct'].mean():.3f} invalid={int(df['invalid'].sum())} ({(time.perf_counter() - t0) / 60:.1f} min)"
            )


if __name__ == "__main__":
    main()
