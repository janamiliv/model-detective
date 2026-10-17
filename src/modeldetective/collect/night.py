"""Unattended overnight collection that keeps Windows awake while it runs.

Uses SetThreadExecutionState (per-process, released automatically on exit) rather than changing
the power plan. Runs the IRIS primary probe first (most important), then the One Token battery.

    uv run python -m modeldetective.collect.night --model qwen3-1.7b --backend cpu \
        --variants Q8_0 Q4_K_M Q2_K
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from datetime import datetime

from modeldetective.collect import one_token_sweep
from modeldetective.collect.sweep import run_sweep

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def stamp(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    keep_awake()
    t0 = time.perf_counter()
    stamp(f"IRIS primary probe: {a.model} {a.backend} variants={a.variants}")
    run_sweep([a.model], a.backend, a.parallel, a.threads, ["short_bits_16"], a.variants)
    stamp(f"One Token battery: {a.model} {a.backend} variants={a.variants}")
    one_token_sweep.run([a.model], a.backend, a.parallel, a.threads, False, a.variants)
    stamp(f"night run finished in {(time.perf_counter() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
