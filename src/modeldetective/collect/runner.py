"""Resumable sampling jobs: one (model variant, probe) cell -> one parquet file.

Columns follow PLAN.md Phase 3: model, variant, probe, prompt_id, run_id, seed, response_raw,
response_parsed, n_tokens, latency, build_hash (+ parse flags and error).
Failed requests are recorded as rows with ``error`` set, never dropped.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from modeldetective.probes.iris import Probe, parse_response
from modeldetective.serve import LlamaClient, LlamaServer, SamplerConfig

# Qwen3 thinking must be off, otherwise the token cap is spent on <think> text.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class Cell:
    model: str  # e.g. "qwen3-0.6b"
    variant: str  # e.g. "f16", "Q4_K_M"
    run_id: int  # independent runs of the same variant get different ids (null control)
    probe: Probe
    n: int

    @property
    def label(self) -> str:
        return f"{self.model}__{self.variant}__run{self.run_id}__{self.probe.id}"


def seed_for(cell: Cell, i: int) -> int:
    """Deterministic and disjoint across pools: (model, variant, run, probe) each get their own range.

    Shared seeds across pools would plant identical strings on both sides of a comparison
    (prereg section 4), so the variant and run are part of the derivation.
    """
    pool = f"{cell.model}|{cell.variant}|{cell.run_id}|{cell.probe.id}"
    return (zlib.crc32(pool.encode()) % 2_000_000) * 1_000_000 + i


def collect_cell(
    cell: Cell,
    server: LlamaServer,
    sampler: SamplerConfig,
    out_dir: Path,
    parallel: int = 8,
    system_prompt: str | None = None,
) -> Path:
    """Sample ``cell.n`` responses from an already-running server. Skips if the file exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cell.label}.parquet"
    if out.exists():
        return out
    build = server.build_info()
    probe = cell.probe
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
        {"role": "user", "content": probe.prompt}
    ]
    samp = SamplerConfig(**{**sampler.__dict__, "max_tokens": probe.max_tokens})

    def one(i: int) -> dict[str, object]:
        seed = seed_for(cell, i)
        t0 = time.perf_counter()
        row: dict[str, object] = {
            "model": cell.model,
            "variant": cell.variant,
            "probe": probe.id,
            "prompt_id": probe.id,
            "run_id": cell.run_id,
            "seed": seed,
            "build_hash": build,
            "temperature": samp.temperature,
            "top_p": samp.top_p,
            "top_k": samp.top_k,
            "min_p": samp.min_p,
            "max_tokens": samp.max_tokens,
        }
        try:
            with LlamaClient(server.base_url) as c:
                r = c.chat(messages, samp, seed=seed, extra=NO_THINK)
            parsed = parse_response(r.text, probe)
            row.update(
                response_raw=r.text,
                response_parsed=parsed.symbols,
                parse_success=parsed.parse_success,
                invalid_rate=parsed.invalid_rate,
                number_value=parsed.number_value,
                n_tokens=r.n_tokens,
                finish_reason=r.finish_reason,
                error=None,
            )
        except Exception as e:
            row.update(
                response_raw=None,
                response_parsed=None,
                parse_success=False,
                invalid_rate=None,
                number_value=None,
                n_tokens=None,
                finish_reason=None,
                error=f"{type(e).__name__}: {e}",
            )
        row["latency"] = time.perf_counter() - t0
        return row

    with cf.ThreadPoolExecutor(parallel) as ex:
        rows = list(ex.map(one, range(cell.n)))
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    return out
