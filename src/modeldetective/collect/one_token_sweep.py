"""Collect the One Token battery per pool, with top-20 logprobs (prereg sections 5-6).

Counts and the language/variant subset come from configs/probes.yaml (family one_token):
English on every variant; ru/zh/ar only on the subset variants.

    uv run python -m modeldetective.collect.one_token_sweep --models qwen3-0.6b --backend cpu --dry-run
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import yaml

from modeldetective.collect.runner import Cell, collect_cell
from modeldetective.collect.sweep import DATA_ROOT, MODELS_CFG, PROBES_CFG, load_sampler, plan_pools
from modeldetective.probes.one_token import battery_probes
from modeldetective.serve import LlamaServer


def one_token_cfg() -> dict:
    cfg = yaml.safe_load(PROBES_CFG.read_text(encoding="utf-8"))
    for p in cfg.get("secondary", []):
        if p.get("family") == "one_token":
            return p
    raise SystemExit("no one_token entry in configs/probes.yaml")


def cells_for_pool(pool, cfg: dict) -> list[tuple[Cell, str]]:  # type: ignore[no-untyped-def]
    subset = set(cfg["subset_variants"])
    out = []
    for probe, system in battery_probes():
        lang = probe.id.rsplit("__", 1)[1]
        if lang != "en" and pool.variant not in subset:
            continue
        n = int(cfg["n_per_cell"][lang])
        out.append((Cell(pool.model, pool.variant, pool.run_id, probe, n), system))
    return out


def run(models: list[str], backend: str, parallel: int, threads: int, dry_run: bool) -> None:
    models_cfg = yaml.safe_load(MODELS_CFG.read_text(encoding="utf-8"))
    cfg = one_token_cfg()
    sampler = load_sampler()
    n_lp = int(cfg.get("logprobs_top_k", 20))
    gpu_kw = (
        {"n_gpu_layers": 0, "device": "none"}
        if backend == "cpu"
        else {"n_gpu_layers": 99, "device": None}
    )
    out_dir = DATA_ROOT / backend / "one_token"
    total = 0
    for key in models:
        for pool in plan_pools(key, models_cfg):
            cells = cells_for_pool(pool, cfg)
            todo = [(c, s) for c, s in cells if not (out_dir / f"{c.label}.parquet").exists()]
            total += sum(c.n for c, _ in cells)
            if dry_run:
                print(
                    f"{pool.model} {pool.variant} run{pool.run_id}: {len(cells)} cells, "
                    f"{sum(c.n for c, _ in cells)} requests, {len(todo)} todo"
                )
                continue
            if not todo:
                continue
            srv = LlamaServer(
                model_path=pool.file,
                port=int(os.environ.get("MD_PORT", "8101")),
                parallel=parallel,
                ctx_size=256 * parallel,
                threads=threads,
                log_path=out_dir / f"server_{pool.model}_{pool.variant}_run{pool.run_id}.log",
                startup_timeout_s=600,
                **gpu_kw,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            with srv:
                print(f"[server] {pool.model} {pool.variant} run{pool.run_id} ({srv.build_info()})")
                for c, system in todo:
                    path = collect_cell(
                        c,
                        srv,
                        sampler,
                        out_dir,
                        parallel=parallel,
                        system_prompt=system,
                        n_logprobs=n_lp,
                    )
                    df = pd.read_parquet(path)
                    print(
                        f"  {c.label}: n={len(df)} errors={int(df['error'].notna().sum())} "
                        f"mean_tokens={df['n_tokens'].mean():.1f}"
                    )
            print(f"  pool done in {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"total requests planned: {total}")


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
