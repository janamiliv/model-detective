"""Detector (a): One Token JSD fingerprint, per pre-registration §8-9, English cells.

Per (model, variant): the ten English battery cells (300 responses per cell per pool). Answers are
normalized (probes/one_token.py); only valid answers enter distributions.

Descriptive: battery JSD (mean over cells) of each full pool vs f16 run 1, next to f16 run 2 vs
f16 run 1 (the null), exactly the One Token paper's distance.

Power: per split (20), each reference cell is split 50/25/25 into train / calibrate / test-ref;
the variant pool's second half is test-sub; the f16 run-2 pool is the null. An episode at budget m
draws m answers from every cell (total queries = 10 m); its statistic is the mean over cells of
JSD(episode answers, train distribution). Threshold = 95th percentile of calibrate episodes.
Reported interval: 2.5-97.5 percentile of power across splits (not a response-level bootstrap;
noted as a simplification for this detector).

    uv run python -m modeldetective.analysis.one_token_analysis --model qwen3-0.6b --backend cuda
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from modeldetective.detect.jsd import jsd
from modeldetective.probes.one_token import load_battery, normalize_answer

BUDGETS = (1, 2, 4, 8, 16)  # per cell; calibrate split is 75 per cell, so m <= 18 is reliable
VARIANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
N_SPLITS, N_EPISODES, ALPHA, SEED = 20, 500, 0.05, 20261014


def load_cells(data: Path, model: str, variant: str, run: int) -> dict[str, list[str]] | None:
    _, tasks = load_battery()
    cells = {}
    for t in tasks:
        p = data / f"{model}__{variant}__run{run}__ot__{t.id}__en.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        ans = [normalize_answer(x, t) for x in df["response_raw"]]
        cells[t.id] = [a for a, cls in ans if cls == "valid"]
    return cells


def validity(data: Path, model: str, variant: str, run: int) -> float:
    _, tasks = load_battery()
    tot = ok = 0
    for t in tasks:
        df = pd.read_parquet(data / f"{model}__{variant}__run{run}__ot__{t.id}__en.parquet")
        cls = [normalize_answer(x, t)[1] for x in df["response_raw"]]
        tot += len(cls)
        ok += sum(c == "valid" for c in cls)
    return ok / tot


def battery_distance(a: dict[str, list[str]], b: dict[str, list[str]], min_n: int = 10) -> float:
    vals = [jsd(a[c], b[c]) for c in a if len(a[c]) >= min_n and len(b.get(c, [])) >= min_n]
    return float(np.mean(vals)) if vals else float("nan")


def _episodes(pool: dict[str, np.ndarray], train: dict[str, Counter], m: int, rng) -> np.ndarray:
    stats = np.zeros(N_EPISODES)
    cells = [c for c in pool if len(pool[c]) >= 1]
    for e in range(N_EPISODES):
        vals = []
        for c in cells:
            arr = pool[c]
            pick = arr[rng.choice(len(arr), size=min(m, len(arr)), replace=False)]
            vals.append(jsd(Counter(pick.tolist()), train[c]))
        stats[e] = np.mean(vals)
    return stats


MIN_VALID = 10  # paper: a cell enters only if both sides have >= 10 valid answers


def usable_cells(ref: dict, null: dict, sub: dict) -> list[str]:
    """Cells where every split part keeps >= MIN_VALID valid answers (ref is cut 50/25/25)."""
    return [
        c
        for c in ref
        if len(ref[c]) >= 4 * MIN_VALID
        and len(null.get(c, [])) >= MIN_VALID
        and len(sub.get(c, [])) >= 2 * MIN_VALID
    ]


def power(ref: dict, null: dict, sub: dict) -> dict:
    cells = usable_cells(ref, null, sub)
    if not cells:
        return {
            "usable_cells": 0,
            "applicable": False,
            "queries_to_detect_80": None,
            "queries_to_detect_95": None,
            "power": [],
            "fpr_null": [],
        }
    ref = {c: ref[c] for c in cells}
    per_split = {m: {"power": [], "fpr_ref": [], "fpr_null": []} for m in BUDGETS}
    for k in range(N_SPLITS):
        rng = np.random.default_rng(SEED + k)
        train, cal, test_ref, test_sub, nullp = {}, {}, {}, {}, {}
        for c, answers in ref.items():
            arr = np.array(answers, dtype=object)
            perm = rng.permutation(len(arr))
            n = len(arr)
            tr, ca, te = perm[: n // 2], perm[n // 2 : 3 * n // 4], perm[3 * n // 4 :]
            train[c] = Counter(arr[tr].tolist())
            cal[c], test_ref[c] = arr[ca], arr[te]
            s = np.array(sub.get(c, []), dtype=object)
            test_sub[c] = s[rng.permutation(len(s))[len(s) // 2 :]]
            nullp[c] = np.array(null.get(c, []), dtype=object)
        for m in BUDGETS:
            thr = np.quantile(_episodes(cal, train, m, rng), 1 - ALPHA)
            per_split[m]["power"].append(float(np.mean(_episodes(test_sub, train, m, rng) > thr)))
            per_split[m]["fpr_ref"].append(float(np.mean(_episodes(test_ref, train, m, rng) > thr)))
            per_split[m]["fpr_null"].append(float(np.mean(_episodes(nullp, train, m, rng) > thr)))
    out = {
        "applicable": True,
        "usable_cells": len(cells),
        "budgets_per_cell": list(BUDGETS),
        "total_queries": [len(cells) * m for m in BUDGETS],
    }
    for key in ("power", "fpr_ref", "fpr_null"):
        out[key] = [float(np.mean(per_split[m][key])) for m in BUDGETS]
    out["power_lo"] = [float(np.percentile(per_split[m]["power"], 2.5)) for m in BUDGETS]
    out["power_hi"] = [float(np.percentile(per_split[m]["power"], 97.5)) for m in BUDGETS]
    reliable = [0.02 <= f <= 0.08 for f in out["fpr_null"]]
    out["reliable"] = reliable
    for target, name in ((0.8, "queries_to_detect_80"), (0.95, "queries_to_detect_95")):
        out[name] = next(
            (
                len(cells) * m
                for m, p, r in zip(BUDGETS, out["power"], reliable, strict=True)
                if r and p >= target
            ),
            None,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", required=True)
    a = ap.parse_args()
    data = Path("data/sweep") / a.backend / "one_token"
    ref, null = load_cells(data, a.model, "f16", 1), load_cells(data, a.model, "f16", 2)
    res = {
        "model": a.model,
        "backend": a.backend,
        "language": "en",
        "null_battery_jsd": battery_distance(null, ref),
        "validity": {
            "f16_run1": validity(data, a.model, "f16", 1),
            "f16_run2": validity(data, a.model, "f16", 2),
        },
        "variants": {},
    }
    print(
        f"{a.model}: null battery JSD (f16 run2 vs run1) = {res['null_battery_jsd']:.3f}",
        flush=True,
    )
    for q in VARIANTS:
        sub = load_cells(data, a.model, q, 1)
        if sub is None:
            continue
        pw = power(ref, null, sub)
        entry = {
            "battery_jsd": battery_distance(sub, ref),
            "validity": validity(data, a.model, q, 1),
            **pw,
        }
        res["variants"][q] = entry
        print(
            f"{q:7s} JSD={entry['battery_jsd']:.3f} valid={entry['validity']:.2f} "
            f"cells={pw['usable_cells']} power={[round(p, 2) for p in pw['power']]} "
            f"fpr_null={[round(p, 3) for p in pw['fpr_null']]} "
            f"q80={pw['queries_to_detect_80']} q95={pw['queries_to_detect_95']}",
            flush=True,
        )
    out = Path("results/one_token") / f"{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
