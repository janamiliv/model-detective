"""Phase 1 baseline reproduction: IRIS Table 8 (q4 vs fp16, Qwen3 0.6B) on our stack.

What IRIS did (arXiv 2607.20860 App. S10.1): Ollama qwen3:0.6b-fp16 vs qwen3:0.6b (default q4),
probes short_bits_16 and single_number_100, 200 responses per label, T=1.0, top_p=1.0,
no system prompt, thinking off; random forest tell rate with 40 stratified splits, alpha 0.05;
an independent second fp16 run as a negative control.

What we do differently (recorded in DECISIONS.md and docs/related_work.md):
- llama-server b11429 on CPU instead of Ollama; our own f16 GGUF and our own Q4_K_M from it.
- Sampler fully untruncated: top_k=0, min_p=0, repeat_penalty=1.0 (IRIS never set top_k,
  so Ollama's default applied and is unrecorded).
- Fixed, recorded seeds per request.

Run:  uv run python -m modeldetective.analysis.phase1_iris [collect|analyze|all]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modeldetective.collect.runner import Cell, collect_cell
from modeldetective.detect.features import extract_features, select_columns
from modeldetective.detect.tell_rate import tell_rate
from modeldetective.probes.iris import IRIS_PROBES, parse_response
from modeldetective.serve import LlamaServer, SamplerConfig

MODEL = "qwen3-0.6b"
MODEL_DIR = Path("models/qwen3-0.6b")
FILES = {"f16": "Qwen3-0.6B-f16.gguf", "Q4_K_M": "Qwen3-0.6B-Q4_K_M.own.gguf"}
PROBES = ["short_bits_16", "single_number_100"]
N_MAIN, N_NULL = 200, 250
DATA_DIR = Path("data/phase1_iris")
RESULTS = Path("results/phase1_iris_repro.json")
SAMPLER = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, repeat_penalty=1.0)
PORT = 8093
PARALLEL = 8

# (variant, run_id, n): run 1 of f16 is the reference, run 2 is the null control
RUNS = [("f16", 1, N_MAIN), ("Q4_K_M", 1, N_MAIN), ("f16", 2, N_NULL)]


def collect() -> None:
    for variant, run_id, n in RUNS:
        cells = [Cell(MODEL, variant, run_id, IRIS_PROBES[p], n) for p in PROBES]
        if all((DATA_DIR / f"{c.label}.parquet").exists() for c in cells):
            print(f"[skip] {variant} run{run_id} already collected")
            continue
        srv = LlamaServer(
            model_path=MODEL_DIR / FILES[variant],
            port=PORT,
            parallel=PARALLEL,
            ctx_size=512 * PARALLEL,
            threads=8,
            log_path=DATA_DIR / f"server_{variant}_run{run_id}.log",
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with srv:
            print(f"[server] {variant} run{run_id} {srv.build_info()}")
            for c in cells:
                out = collect_cell(c, srv, SAMPLER, DATA_DIR, parallel=PARALLEL)
                df = pd.read_parquet(out)
                print(
                    f"  {c.label}: n={len(df)} errors={df['error'].notna().sum()} "
                    f"parse_ok={df['parse_success'].mean():.3f} "
                    f"mean_tokens={df['n_tokens'].mean():.1f}"
                )


def _features(df: pd.DataFrame, probe_id: str) -> pd.DataFrame:
    probe = IRIS_PROBES[probe_id]
    rows = []
    for text in df["response_raw"].fillna(""):
        rows.append(extract_features(text, parse_response(text, probe), probe))
    return pd.DataFrame(rows)


def _load(variant: str, run_id: int, probe_id: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{MODEL}__{variant}__run{run_id}__{probe_id}.parquet")


def analyze() -> dict:
    out: dict = {"model": MODEL, "sampler": SAMPLER.__dict__, "cells": {}}
    for probe_id in PROBES:
        ref, sub, null = (_load(v, r, probe_id) for v, r, _ in RUNS)
        f_ref, f_sub, f_null = (_features(d, probe_id) for d in (ref, sub, null))
        cols_all = list(f_ref.columns)
        res: dict = {
            "n_ref": len(ref),
            "n_sub": len(sub),
            "n_null": len(null),
            "parse_success": {
                "ref": float(ref["parse_success"].mean()),
                "sub": float(sub["parse_success"].mean()),
                "null": float(null["parse_success"].mean()),
            },
            "mean_tokens": {
                "ref": float(ref["n_tokens"].mean()),
                "sub": float(sub["n_tokens"].mean()),
                "null": float(null["n_tokens"].mean()),
            },
            "errors": int(ref["error"].notna().sum() + sub["error"].notna().sum()),
        }
        for feat_set, content_only in (("full", False), ("content_only", True)):
            cols = select_columns(cols_all, content_only=content_only)
            x_ref = f_ref[cols].to_numpy(float)
            r_sub = tell_rate(x_ref, f_sub[cols].to_numpy(float))
            r_null = tell_rate(x_ref, f_null[cols].to_numpy(float))
            res[feat_set] = {
                "n_features": len(cols),
                "q4_vs_f16": r_sub.as_dict(),
                "f16_vs_f16_null": r_null.as_dict(),
            }
            print(
                f"{probe_id:18s} {feat_set:12s} feats={len(cols):3d} | "
                f"q4: q={r_sub.q_mean:.2f} [{r_sub.q_lo:.2f},{r_sub.q_hi:.2f}] "
                f"auroc={r_sub.auroc_mean:.2f} floor={'yes' if r_sub.clears_floor else 'no '} | "
                f"null: q={r_null.q_mean:.2f} [{r_null.q_lo:.2f},{r_null.q_hi:.2f}] "
                f"auroc={r_null.auroc_mean:.2f}"
            )
        out["cells"][probe_id] = res
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(out, indent=2, default=_json_default), encoding="utf-8", newline="\n"
    )
    print(f"wrote {RESULTS}")
    return out


def _json_default(o: object) -> object:
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("collect", "all"):
        collect()
    if cmd in ("analyze", "all"):
        analyze()
