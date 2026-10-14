"""Phase 2: HF checkpoint -> f16 GGUF -> K-quant variants, with a manifest per file.

Everything is pinned: the HF revision comes from configs/models.yaml (or models/<m>/hf/.revision),
the llama.cpp build is the one under tools/llama.cpp (binary for quantize, source for convert).
Each step is skipped if its output exists, so the script is resumable.

Run:  uv run python -m modeldetective.quantize.pipeline qwen3-0.6b [qwen3-1.7b ...] [--smoke]
Outputs: models/<m>/*.gguf (local only), models/<m>/manifests/*.json (local),
         results/variants_table.md (committed summary).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

LLAMA_BIN = Path("tools/llama.cpp/bin")
LLAMA_SRC = Path("tools/llama.cpp/src")
CONFIG = Path("configs/models.yaml")
TABLE = Path("results/variants_table.md")
QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
SMOKE_PROMPT = "The capital of Serbia is"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def llama_build() -> str:
    out = subprocess.run(
        [str(LLAMA_BIN / "llama-cli.exe"), "--version"], capture_output=True, text=True
    )
    m = re.search(r"build (\d+), commit ([0-9a-f]+)", out.stdout + out.stderr)
    return f"b{m.group(1)}-{m.group(2)}" if m else "unknown"


def gguf_param_count(path: Path) -> int | None:
    try:
        from gguf import GGUFReader  # type: ignore[import-untyped]

        r = GGUFReader(str(path))
        f = r.fields.get("general.parameter_count")
        if f is not None:
            return int(f.parts[f.data[0]][0])
        return int(sum(int(t.n_elements) for t in r.tensors))
    except Exception:
        return None


def write_manifest(
    path: Path, kind: str, model_key: str, cfg: dict, build: str, command: list[str], extra: dict
) -> dict:
    n_params = gguf_param_count(path)
    size = path.stat().st_size
    man = {
        "model": model_key,
        "hf_id": cfg["hf_id"],
        "hf_revision": cfg["hf_revision"],
        "kind": kind,  # "reference" or quant type
        "file": path.name,
        "size_bytes": size,
        "sha256": sha256(path),
        "n_params": n_params,
        "bpw_from_size": round(size * 8 / n_params, 3) if n_params else None,
        "llama_cpp_build": build,
        "command": command,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }
    mdir = path.parent / "manifests"
    mdir.mkdir(exist_ok=True)
    (mdir / f"{path.stem}.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8", newline="\n"
    )
    return man


def convert_f16(model_key: str, cfg: dict, mdir: Path, build: str) -> tuple[Path, dict | None]:
    stem = cfg["hf_id"].split("/")[-1]
    out = mdir / f"{stem}-f16.gguf"
    cmd = [
        sys.executable,
        str(LLAMA_SRC / "convert_hf_to_gguf.py"),
        str(mdir / "hf"),
        "--outtype",
        "f16",
        "--outfile",
        str(out),
    ]
    if out.exists():
        print(f"[skip] {out.name} exists")
        return out, None
    print(f"[convert] {model_key} -> {out.name}")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    man = write_manifest(
        out, "reference", model_key, cfg, build, cmd, {"seconds": round(time.perf_counter() - t0)}
    )
    return out, man


def quantize(
    model_key: str, cfg: dict, f16: Path, qtype: str, build: str
) -> tuple[Path, dict | None]:
    out = f16.with_name(f16.name.replace("-f16.gguf", f"-{qtype}.own.gguf"))
    cmd = [str(LLAMA_BIN / "llama-quantize.exe"), str(f16), str(out), qtype]
    if out.exists():
        if not (out.parent / "manifests" / f"{out.stem}.json").exists():
            man = write_manifest(
                out,
                qtype,
                model_key,
                cfg,
                build,
                cmd,
                {"note": "pre-existing file, same pinned build", "imatrix": False},
            )
            return out, man
        print(f"[skip] {out.name} exists")
        return out, None
    print(f"[quantize] {out.name}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    log = res.stdout + res.stderr
    m = re.search(r"quant size\s*=\s*([\d.]+) MiB \(([\d.]+) BPW\)", log)
    extra = {
        "seconds": round(time.perf_counter() - t0),
        "bpw_reported": float(m.group(2)) if m else None,
        "imatrix": False,
    }
    man = write_manifest(out, qtype, model_key, cfg, build, cmd, extra)
    return out, man


def smoke(path: Path, backend: str) -> dict:
    cmd = [
        str(LLAMA_BIN / "llama-completion.exe"),
        "-m",
        str(path),
        "-n",
        "24",
        "--temp",
        "0",
        "-no-cnv",
        "-p",
        SMOKE_PROMPT,
    ]
    cmd += ["-ngl", "99"] if backend == "cuda" else ["--device", "none", "-ngl", "0"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    text = res.stdout.strip().replace(SMOKE_PROMPT, "", 1).strip()
    return {"ok": res.returncode == 0 and len(text) > 0, "text": text[:120], "backend": backend}


def load_manifests(mdir: Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((mdir / "manifests").glob("*.json"))
    ]


def write_table(model_keys: list[str]) -> None:
    rows = [
        "# GGUF variants",
        "",
        "Generated by `modeldetective.quantize.pipeline`. "
        "Files are local only; hashes are the ground truth.",
        "",
    ]
    rows += [
        "| Model | Variant | Size (MiB) | Bits/weight (file) | Params | SHA256 (first 16) | Build | Smoke |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = {"reference": -1, **{q: i for i, q in enumerate(QUANTS)}}
    for key in model_keys:
        mdir = Path("models") / key
        if not (mdir / "manifests").exists():
            continue
        for man in sorted(load_manifests(mdir), key=lambda m: order.get(m["kind"], 99)):
            bpw = man.get("bpw_from_size")  # one basis for every row: file bits / parameters
            smoke_s = man.get("smoke", {}).get("text", "")[:40].replace("|", "/").replace("\n", " ")
            rows.append(
                f"| {key} | {man['kind'] if man['kind'] != 'reference' else 'f16 (reference)'} | "
                f"{man['size_bytes'] / 2**20:,.0f} | {bpw if bpw else ''} | "
                f"{man['n_params'] / 1e6:,.0f}M | `{man['sha256'][:16]}` | {man['llama_cpp_build']} | {smoke_s} |"
            )
    TABLE.parent.mkdir(exist_ok=True)
    TABLE.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {TABLE}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="keys from configs/models.yaml, e.g. qwen3-0.6b")
    ap.add_argument("--quants", nargs="*", default=QUANTS)
    ap.add_argument("--smoke", action="store_true", help="greedy smoke test of every file")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    cfgs = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["models"]
    build = llama_build()
    print(f"llama.cpp build: {build}")
    for key in args.models:
        cfg = dict(cfgs[key])
        mdir = Path("models") / key
        rev_file = mdir / "hf" / ".revision"
        if rev_file.exists():
            cfg["hf_revision"] = rev_file.read_text().strip()
        if not (mdir / "hf" / "config.json").exists():
            raise SystemExit(f"{key}: HF checkpoint missing at {mdir / 'hf'}; download it first")
        f16, _ = convert_f16(key, cfg, mdir, build)
        if not (mdir / "manifests" / f"{f16.stem}.json").exists():
            write_manifest(f16, "reference", key, cfg, build, ["(pre-existing)"], {})
        for q in args.quants:
            quantize(key, cfg, f16, q, build)
        if args.smoke:
            for man_path in sorted((mdir / "manifests").glob("*.json")):
                man = json.loads(man_path.read_text(encoding="utf-8"))
                gguf = mdir / man["file"]
                man["smoke"] = smoke(gguf, args.backend)
                man_path.write_text(json.dumps(man, indent=2), encoding="utf-8", newline="\n")
                flag = "ok " if man["smoke"]["ok"] else "BAD"
                print(f"[smoke {flag}] {gguf.name}: {man['smoke']['text'][:70]!r}")
    write_table(list(cfgs))


if __name__ == "__main__":
    main()
