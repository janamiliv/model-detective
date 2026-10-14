"""Phase 3 sweep: collect every (model, variant, run, probe) pool listed in configs/probes.yaml.

Pools per model: f16 run 1 (reference), f16 run 2 (null control), and run 1 of every quantized
variant. One llama-server start per (model, variant, run); cells are skipped if their parquet
exists, so the sweep is resumable. Backend is fixed per invocation (prereg section 4).

    uv run python -m modeldetective.collect.sweep --models qwen3-0.6b --backend cpu --dry-run
    uv run python -m modeldetective.collect.sweep --models qwen3-0.6b --backend cpu
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from modeldetective.collect.runner import Cell, collect_cell
from modeldetective.probes.iris import IRIS_PROBES
from modeldetective.serve import LlamaServer, SamplerConfig

MODELS_CFG = Path("configs/models.yaml")
PROBES_CFG = Path("configs/probes.yaml")
SAMPLER_CFG = Path("configs/sampler.yaml")
DATA_ROOT = Path("data/sweep")
QUANT_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


@dataclass(frozen=True)
class Pool:
    model: str
    variant: str
    run_id: int
    file: Path


def load_sampler() -> SamplerConfig:
    s = yaml.safe_load(SAMPLER_CFG.read_text(encoding="utf-8"))["primary"]
    return SamplerConfig(
        temperature=s["temperature"],
        top_p=s["top_p"],
        top_k=s["top_k"],
        min_p=s["min_p"],
        repeat_penalty=s["repeat_penalty"],
    )


def iris_probe_specs(probes_cfg: dict) -> list[tuple[str, int]]:
    """(probe_id, n_per_pool) for the IRIS-family probes, primary first."""
    out = [(probes_cfg["primary"]["probe"], int(probes_cfg["primary"]["n_per_pool"]))]
    for p in probes_cfg.get("secondary", []):
        if p.get("family") == "iris":
            out.append((p["probe"], int(p["n_per_pool"])))
    return out


def plan_pools(model_key: str, models_cfg: dict, variants: list[str] | None = None) -> list[Pool]:
    m = models_cfg["models"][model_key]
    mdir = Path(m.get("local_dir", f"models/{model_key}"))
    pools = [
        Pool(model_key, "f16", 1, mdir / m["reference"]["file"]),
        Pool(model_key, "f16", 2, mdir / m["reference"]["file"]),
    ]
    for q in QUANT_ORDER:
        if q in m.get("variants", {}) and (variants is None or q in variants):
            pools.append(Pool(model_key, q, 1, mdir / m["variants"][q]["file"]))
    return pools


def plan_cells(pool: Pool, probe_specs: list[tuple[str, int]]) -> list[Cell]:
    return [Cell(pool.model, pool.variant, pool.run_id, IRIS_PROBES[p], n) for p, n in probe_specs]


def cell_path(cell: Cell, backend: str) -> Path:
    return DATA_ROOT / backend / f"{cell.label}.parquet"


def run_sweep(
    models: list[str],
    backend: str,
    parallel: int = 8,
    threads: int = 8,
    probes: list[str] | None = None,
    variants: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    models_cfg = yaml.safe_load(MODELS_CFG.read_text(encoding="utf-8"))
    probes_cfg = yaml.safe_load(PROBES_CFG.read_text(encoding="utf-8"))
    specs = [s for s in iris_probe_specs(probes_cfg) if probes is None or s[0] in probes]
    sampler = load_sampler()
    gpu_kw = (
        {"n_gpu_layers": 0, "device": "none"}
        if backend == "cpu"
        else {"n_gpu_layers": 99, "device": None}
    )
    out_dir = DATA_ROOT / backend
    total_req = done_req = 0
    for key in models:
        for pool in plan_pools(key, models_cfg, variants):
            cells = plan_cells(pool, specs)
            todo = [c for c in cells if not cell_path(c, backend).exists()]
            total_req += sum(c.n for c in cells)
            done_req += sum(c.n for c in cells if c not in todo)
            if dry_run:
                for c in cells:
                    state = "done" if c not in todo else "todo"
                    print(f"[{state}] {c.label}  n={c.n}")
                continue
            if not todo:
                print(f"[skip] {pool.model} {pool.variant} run{pool.run_id}: all cells collected")
                continue
            if not pool.file.exists():
                raise SystemExit(f"model file missing: {pool.file}")
            srv = LlamaServer(
                model_path=pool.file,
                port=int(os.environ.get("MD_PORT", "8100")),
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
                for c in todo:
                    t1 = time.perf_counter()
                    path = collect_cell(c, srv, sampler, out_dir, parallel=parallel)
                    df = pd.read_parquet(path)
                    rate = len(df) / (time.perf_counter() - t1) * 60
                    print(
                        f"  {c.label}: n={len(df)} errors={int(df['error'].notna().sum())} "
                        f"parse_ok={df['parse_success'].mean():.3f} "
                        f"mean_tokens={df['n_tokens'].mean():.1f} ({rate:.0f}/min)"
                    )
            print(f"  pool done in {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"requests: {done_req}/{total_req} collected for {models} on {backend}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--probes", nargs="*", default=None, help="subset of IRIS probe ids")
    ap.add_argument("--variants", nargs="*", default=None, help="subset of quant types")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run_sweep(a.models, a.backend, a.parallel, a.threads, a.probes, a.variants, a.dry_run)


if __name__ == "__main__":
    main()
