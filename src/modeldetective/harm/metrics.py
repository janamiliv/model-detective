"""Accuracy, flip rate and bootstrap CIs for the detection-harm comparison (prereg section 10).

flip rate = fraction of questions whose correctness differs between the variant and the f16
reference (correct->wrong plus wrong->correct). The f16-vs-f16 flip rate (two greedy runs of the
reference) is the null; with deterministic greedy decoding it should be 0 or near 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarmResult:
    n: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    accuracy_drop: float  # reference accuracy minus variant accuracy
    accuracy_drop_ci: tuple[float, float]
    flip_rate: float
    flip_rate_ci: tuple[float, float]
    flips_correct_to_wrong: int
    flips_wrong_to_correct: int
    invalid_answers: int  # responses that were not a letter A-D

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "accuracy_ci95": list(self.accuracy_ci),
            "accuracy_drop": self.accuracy_drop,
            "accuracy_drop_ci95": list(self.accuracy_drop_ci),
            "flip_rate": self.flip_rate,
            "flip_rate_ci95": list(self.flip_rate_ci),
            "flips_correct_to_wrong": self.flips_correct_to_wrong,
            "flips_wrong_to_correct": self.flips_wrong_to_correct,
            "invalid_answers": self.invalid_answers,
        }


def _ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def harm(
    ref_correct: np.ndarray,
    var_correct: np.ndarray,
    var_invalid: int = 0,
    n_boot: int = 1000,
    seed: int = 20261014,
) -> HarmResult:
    ref_correct = np.asarray(ref_correct, dtype=float)
    var_correct = np.asarray(var_correct, dtype=float)
    if ref_correct.shape != var_correct.shape:
        raise ValueError("reference and variant must cover the same questions in the same order")
    rng = np.random.default_rng(seed)
    flips = (ref_correct != var_correct).astype(float)
    drop = ref_correct - var_correct  # per-question; mean = accuracy drop
    return HarmResult(
        n=len(var_correct),
        accuracy=float(var_correct.mean()),
        accuracy_ci=_ci(var_correct, n_boot, rng),
        accuracy_drop=float(drop.mean()),
        accuracy_drop_ci=_ci(drop, n_boot, rng),
        flip_rate=float(flips.mean()),
        flip_rate_ci=_ci(flips, n_boot, rng),
        flips_correct_to_wrong=int(((ref_correct == 1) & (var_correct == 0)).sum()),
        flips_wrong_to_correct=int(((ref_correct == 0) & (var_correct == 1)).sum()),
        invalid_answers=int(var_invalid),
    )
