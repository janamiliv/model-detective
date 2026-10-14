"""Collect the MET-style natural prompts per pool (configs/met_prompts.yaml; prereg section 5).

40 completions per prompt per pool, 50-token cap, no system prompt, raw text kept. These feed
the Hamming-MMD detector (Phase 4, detector b).

    uv run python -m modeldetective.collect.met_sweep --models qwen3-0.6b --backend cpu --dry-run
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import yaml

from modeldetective.collect.runner import Cell, collect_cell
from modeldetective.collect.sweep import DATA_ROOT, MODELS_CFG, PROBES_CFG, load_sampler, plan_pools
from modeldetective.probes.iris import Probe
from modeldetective.serve import LlamaServer

MET_FILE = Path("configs/met_prompts.yaml")


def met_probes() -> list[Probe]:
    d = yaml.safe_load(MET_FILE.read_text(encoding="utf-8"))
    cap = int(d["max_tokens"])
    return [Probe(p["id"], p["prompt"], "", 1, "raw", cap) for p in d["prompts"]]


def n_per_prompt() -> int:
    cfg = yaml.safe_load(PROBES_CFG.read_text(encoding="utf-8"))
    for p in cfg.get("secondary", []):
        if p.get("family") == "met_natural":
            return int(p["n_per_prompt_per_pool"])
    raise SystemExit("no met_natural entry in configs/probes.yaml")


def run(models: list[str], backend: str, parallel: int, threads: int, dry_run: bool) -> None:
    models_cfg = yaml.safe_load(MODELS_CFG.read_text(encoding="utf-8"))
    probes = met_probes()
    n = n_per_prompt()
    sampler = load_sampler()
    gpu_kw = (
        {"n_gpu_layers": 0, "device": "none"}
        if backend == "cpu"
        else {"n_gpu_layers": 99, "device": None}
    )
    out_dir = DATA_ROOT / backend / "met"
    for key in models:
        for pool in plan_pools(key, models_cfg):
            cells = [Cell(pool.model, pool.variant, pool.run_id, p, n) for p in probes]
            todo = [c for c in cells if not (out_dir / f"{c.label}.parquet").exists()]
            if dry_run:
                print(
                    f"{pool.model} {pool.variant} run{pool.run_id}: {len(cells)} prompts x {n}, "
                    f"{len(todo)} todo"
                )
                continue
            if not todo:
                continue
            srv = LlamaServer(
                model_path=pool.file,
                port=int(os.environ.get("MD_PORT", "8102")),
                parallel=parallel,
                ctx_size=512 * parallel,
                threads=threads,
                log_path=out_dir / f"server_{pool.model}_{pool.variant}_run{pool.run_id}.log",
                startup_timeout_s=600,
                **gpu_kw,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            with srv:
                print(f"[server] {pool.model} {pool.variant} run{pool.run_id} ({srv.build_info()})")
                errors = 0
                for c in todo:
                    df = pd.read_parquet(collect_cell(c, srv, sampler, out_dir, parallel=parallel))
                    errors += int(df["error"].notna().sum())
                print(
                    f"  {len(todo)} prompts done, errors={errors}, "
                    f"{(time.perf_counter() - t0) / 60:.1f} min"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.models, a.backend, a.parallel, a.threads, a.dry_run)


if __name__ == "__main__":
    main()
