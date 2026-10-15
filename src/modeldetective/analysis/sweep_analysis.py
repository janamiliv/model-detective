"""Phase 4 analysis: detector (c) tell rate + pre-registered power curves per variant and probe.

For each (model, probe, variant):
  pools: reference = f16 run 1, null = f16 run 2, substitute = variant run 1
  repeat over N_SPLITS random splits (prereg section 9):
    reference -> train 50% / calibrate 25% / test-ref 25%; substitute -> train 50% / test-sub 50%
    RandomForest on (train-ref vs train-sub) with content-only features (primary) and full (secondary)
    score s(y) = -log P(reference | y) on calibrate, test-ref, null, test-sub
    power curve over budgets, FPR on test-ref and on null, queries-to-detect at 80% / 95%
  aggregate: mean power per budget across splits; CI = mean of per-split bootstrap bounds
Outputs results/sweep/<backend>/<model>__<probe>.json and a markdown table.

    uv run python -m modeldetective.analysis.sweep_analysis --models qwen3-0.6b --backend cpu
    uv run python -m modeldetective.analysis.sweep_analysis --data data/phase1_iris --models qwen3-0.6b --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from modeldetective.detect.features import extract_features, select_columns
from modeldetective.detect.power import power_curve_from_scores, split_indices
from modeldetective.probes.iris import IRIS_PROBES, parse_response

QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
N_SPLITS = 20
SEED = 20261014


def load_pool(
    data_dir: Path, model: str, variant: str, run_id: int, probe: str
) -> pd.DataFrame | None:
    p = data_dir / f"{model}__{variant}__run{run_id}__{probe}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def features(df: pd.DataFrame, probe_id: str) -> pd.DataFrame:
    probe = IRIS_PROBES[probe_id]
    texts = df["response_raw"].fillna("")
    return pd.DataFrame([extract_features(t, parse_response(t, probe), probe) for t in texts])


FPR_BAND = (0.02, 0.08)  # prereg section 9: null FPR must stay here at nominal alpha 0.05


def pool_quality(df: pd.DataFrame) -> dict:
    vc = df["response_raw"].value_counts()
    return {
        "n": len(df),
        "errors": int(df["error"].notna().sum()),
        "parse_success": float(df["parse_success"].mean()),
        "distinct_fraction": float(df["response_raw"].nunique() / max(len(df), 1)),
        "top1_fraction": float(vc.iloc[0] / len(df)) if len(vc) else 0.0,
        "mean_tokens": float(df["n_tokens"].mean()),
    }


def degenerate(q: dict) -> bool:
    """Deviation 2026-10-15 (prereg Deviations section): broken output means parse failure or a
    single dominant string, not merely repetition. Repetition is itself a quantization signal."""
    return q["parse_success"] < 0.5 or q["top1_fraction"] > 0.5


def analyze_variant(
    f_ref: pd.DataFrame,
    f_null: pd.DataFrame,
    f_sub: pd.DataFrame,
    cols: list[str],
    n_splits: int,
    n_episodes: int,
    n_boot: int,
    n_estimators: int,
) -> dict:
    x_ref, x_null, x_sub = (f[cols].to_numpy(float) for f in (f_ref, f_null, f_sub))
    curves = []
    for k in range(n_splits):
        rng = np.random.default_rng(SEED + k)
        tr_r, cal, te_r = split_indices(len(x_ref), (0.5, 0.25, 0.25), rng)
        tr_s, te_s = split_indices(len(x_sub), (0.5, 0.5), rng)
        clf = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED + k,
                n_jobs=-1,
            ),
        )
        x_tr = np.vstack([x_ref[tr_r], x_sub[tr_s]])
        y_tr = np.r_[np.zeros(len(tr_r), int), np.ones(len(tr_s), int)]
        clf.fit(x_tr, y_tr)

        def score(x: np.ndarray, model=clf) -> np.ndarray:
            return -np.log(np.clip(model.predict_proba(x)[:, 0], 1e-12, 1.0))

        pc = power_curve_from_scores(
            score(x_ref[cal]),
            score(x_ref[te_r]),
            score(x_null),
            score(x_sub[te_s]),
            n_episodes=n_episodes,
            n_boot=n_boot,
            seed=SEED + k,
        )
        curves.append(pc)
    budgets = list(curves[0].budgets)
    power = np.array([c.power for c in curves])
    out = {
        "budgets": budgets,
        "power_mean": power.mean(axis=0).tolist(),
        "power_sd_across_splits": power.std(axis=0, ddof=1).tolist()
        if n_splits > 1
        else [0.0] * len(budgets),
        "power_ci95_lo": np.mean([c.power_lo for c in curves], axis=0).tolist(),
        "power_ci95_hi": np.mean([c.power_hi for c in curves], axis=0).tolist(),
        "fpr_ref_mean": np.mean([c.fpr_ref for c in curves], axis=0).tolist(),
        "fpr_null_mean": np.mean([c.fpr_null for c in curves], axis=0).tolist(),
        "n_splits": n_splits,
        "n_features": len(cols),
    }
    pm = out["power_mean"]
    n_cal = round(0.25 * len(x_ref))
    # A budget is reliable only if the null FPR stays in the pre-registered band and the
    # calibration pool is large enough for distinct episodes (m <= n_calibrate / 4).
    reliable = [
        FPR_BAND[0] <= f <= FPR_BAND[1] and m <= n_cal // 4
        for m, f in zip(budgets, out["fpr_null_mean"], strict=True)
    ]
    out["n_calibrate"] = n_cal
    out["reliable"] = reliable
    out["queries_to_detect_80"] = next(
        (m for m, p, r in zip(budgets, pm, reliable, strict=True) if r and p >= 0.8), None
    )
    out["queries_to_detect_95"] = next(
        (m for m, p, r in zip(budgets, pm, reliable, strict=True) if r and p >= 0.95), None
    )
    out["single_query_power"] = pm[0]
    return out


def run(
    data_dir: Path,
    out_dir: Path,
    models: list[str],
    probes: list[str],
    n_splits: int = N_SPLITS,
    n_episodes: int = 1000,
    n_boot: int = 200,
    n_estimators: int = 300,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        for probe in probes:
            ref, null = (
                load_pool(data_dir, model, "f16", 1, probe),
                load_pool(data_dir, model, "f16", 2, probe),
            )
            if ref is None or null is None:
                print(f"[skip] {model} {probe}: reference or null pool missing")
                continue
            f_ref, f_null = features(ref, probe), features(null, probe)
            all_cols = list(f_ref.columns)
            result = {
                "model": model,
                "probe": probe,
                "data_dir": str(data_dir),
                "quality": {"f16_run1": pool_quality(ref), "f16_run2": pool_quality(null)},
                "variants": {},
            }
            for q in QUANTS:
                sub = load_pool(data_dir, model, q, 1, probe)
                if sub is None:
                    continue
                f_sub = features(sub, probe)
                qual = pool_quality(sub)
                entry = {"quality": qual, "degenerate": degenerate(qual)}
                for name, content_only in (("content_only", True), ("full", False)):
                    cols = select_columns(all_cols, content_only=content_only)
                    entry[name] = analyze_variant(
                        f_ref, f_null, f_sub, cols, n_splits, n_episodes, n_boot, n_estimators
                    )
                    r = entry[name]
                    print(
                        f"{model} {probe:18s} {q:7s} {name:12s} "
                        f"P(m=1)={r['single_query_power']:.2f} "
                        f"m80={r['queries_to_detect_80']} m95={r['queries_to_detect_95']} "
                        f"fpr_null(m=1..256)={min(r['fpr_null_mean']):.3f}-{max(r['fpr_null_mean']):.3f}"
                        f"{'  DEGENERATE' if entry['degenerate'] else ''}"
                    )
                result["variants"][q] = entry
            path = out_dir / f"{model}__{probe}.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
            print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--data", default=None, help="override data dir (default data/sweep/<backend>)")
    ap.add_argument("--probes", nargs="*", default=["short_bits_16"])
    ap.add_argument("--smoke", action="store_true", help="few splits/episodes for a quick check")
    a = ap.parse_args()
    data_dir = Path(a.data) if a.data else Path("data/sweep") / a.backend
    out_dir = Path("results/sweep") / (Path(a.data).name if a.data else a.backend)
    kw = dict(n_splits=3, n_episodes=200, n_boot=20, n_estimators=100) if a.smoke else {}
    run(data_dir, out_dir, a.models, a.probes, **kw)


if __name__ == "__main__":
    main()
