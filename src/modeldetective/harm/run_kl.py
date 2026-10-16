"""Run the KL-divergence measurement for one model: save f16 logits once, then KL per variant.

16 chunks of 512 tokens from configs/kl_text.txt (~8k tokens). The base-logit file is about
n_tokens x vocab x 2 bytes (~2.5 GB for Qwen3), so it is written to data/ and deleted after use.

    uv run python -m modeldetective.harm.run_kl --model qwen3-0.6b --backend cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from modeldetective.collect.sweep import MODELS_CFG, QUANT_ORDER
from modeldetective.harm.kl import kl_against_reference, save_reference_logits

TEXT = Path("configs/kl_text.txt")
CHUNKS = 16
CTX = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    a = ap.parse_args()
    m = yaml.safe_load(MODELS_CFG.read_text(encoding="utf-8"))["models"][a.model]
    mdir = Path(m["local_dir"])
    base = Path("data/harm") / f"kl_base_{a.model}.bin"
    base.parent.mkdir(parents=True, exist_ok=True)
    if not base.exists():
        print("saving f16 reference logits ...", flush=True)
        save_reference_logits(mdir / m["reference"]["file"], TEXT, base, a.backend, CTX, CHUNKS)
    out: dict = {"model": a.model, "backend": a.backend, "chunks": CHUNKS, "ctx": CTX}
    # f16 against itself is the null (should be ~0; nonzero only from backend non-determinism)
    targets = [("f16", m["reference"]["file"])] + [
        (q, m["variants"][q]["file"]) for q in QUANT_ORDER if q in m["variants"]
    ]
    for name, f in targets:
        r = kl_against_reference(mdir / f, TEXT, base, a.backend, CTX, CHUNKS)
        out[name] = r
        print(f"{name:7s} " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    res = Path("results/harm") / f"{a.model}_kl.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    base.unlink()
    print(f"wrote {res}; deleted {base}")


if __name__ == "__main__":
    main()
