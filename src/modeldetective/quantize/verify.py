"""Verify every local model file against its manifest (size and SHA256).

Run after moving model files or before a long collection run:
    uv run python -m modeldetective.quantize.verify [qwen3-0.6b ...]
Exit code 1 if anything is missing or mismatched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from modeldetective.quantize.pipeline import sha256


def verify(model_keys: list[str]) -> bool:
    ok = True
    for key in model_keys:
        mdir = Path("models") / key
        for mp in sorted((mdir / "manifests").glob("*.json")):
            man = json.loads(mp.read_text(encoding="utf-8"))
            f = mdir / man["file"]
            if not f.exists():
                print(f"MISSING  {f}")
                ok = False
                continue
            size = f.stat().st_size
            if size != man["size_bytes"]:
                print(f"SIZE     {f.name}: {size} != {man['size_bytes']}")
                ok = False
                continue
            t0 = time.perf_counter()
            h = sha256(f)
            status = "ok" if h == man["sha256"] else "HASH MISMATCH"
            if h != man["sha256"]:
                ok = False
            print(
                f"{status:14s} {f.name:36s} {size / 2**20:8,.0f} MiB  {time.perf_counter() - t0:5.1f}s"
            )
    return ok


def main() -> None:
    keys = sys.argv[1:] or sorted(
        p.name for p in Path("models").iterdir() if (p / "manifests").exists()
    )
    good = verify(keys)
    print("ALL OK" if good else "FAILURES FOUND")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
