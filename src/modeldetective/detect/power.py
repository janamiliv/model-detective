"""Power analysis and queries-to-detect, exactly as pre-registered (docs/prereg.md section 9).

Inputs are per-response scores (detector c: s(y) = -log P(reference | y)) or any per-response
statistic whose episode statistic is the mean over m responses. For set-level detectors (JSD,
MMD) pass an ``episode_stat`` callable instead.

Procedure per split:
  reference pool -> train 50% / calibrate 25% / test-ref 25% (stratified by nothing: i.i.d.)
  null pool      -> scored only
  variant pool   -> train 50% (classifier only) / test-sub 50%
For each budget m: draw episodes (without replacement inside, with replacement across) from
calibrate -> threshold at the 1 - alpha quantile; from test-ref and null -> false-positive rate;
from test-sub -> power. CIs: percentile bootstrap over the underlying responses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def episode_means(
    scores: np.ndarray, m: int, n_episodes: int, rng: np.random.Generator
) -> np.ndarray:
    """Means of n_episodes episodes of m responses drawn without replacement inside each episode."""
    n = len(scores)
    if n == 0:
        return np.full(n_episodes, np.nan)
    m_eff = min(m, n)
    idx = np.stack([rng.choice(n, size=m_eff, replace=False) for _ in range(n_episodes)])
    return scores[idx].mean(axis=1)


@dataclass
class PowerCurve:
    budgets: list[int]
    power: list[float]
    power_lo: list[float]
    power_hi: list[float]
    fpr_ref: list[float]  # held-out reference (same pool as calibration, disjoint split)
    fpr_null: list[float]  # independent null pool
    alpha: float
    n_episodes: int
    queries_to_detect_80: int | None = None
    queries_to_detect_95: int | None = None
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "budgets": self.budgets,
            "power": self.power,
            "power_ci95": list(zip(self.power_lo, self.power_hi, strict=True)),
            "fpr_ref": self.fpr_ref,
            "fpr_null": self.fpr_null,
            "alpha": self.alpha,
            "n_episodes": self.n_episodes,
            "queries_to_detect_80": self.queries_to_detect_80,
            "queries_to_detect_95": self.queries_to_detect_95,
            **self.notes,
        }


def _first_budget(budgets: list[int], power: list[float], target: float) -> int | None:
    for m, p in zip(budgets, power, strict=True):
        if p >= target:
            return m
    return None


def power_curve_from_scores(
    s_calibrate: np.ndarray,
    s_test_ref: np.ndarray,
    s_null: np.ndarray,
    s_test_sub: np.ndarray,
    alpha: float = 0.05,
    budgets: tuple[int, ...] = BUDGETS,
    n_episodes: int = 1000,
    n_boot: int = 200,
    seed: int = 20261014,
    episode_stat: Callable[[np.ndarray, int, int, np.random.Generator], np.ndarray] = episode_means,
) -> PowerCurve:
    """Power, FPRs and queries-to-detect for one split of already-computed per-response scores.

    Higher score = more "substitute-like". Threshold is the (1 - alpha) quantile of calibrate
    episode statistics. Bootstrap resamples the underlying response pools, then recomputes
    threshold and power, so the CI reflects response-level uncertainty (prereg section 9).
    """
    rng = np.random.default_rng(seed)
    bl, power, lo, hi, fpr_r, fpr_n = [], [], [], [], [], []
    for m in budgets:
        thr = np.quantile(episode_stat(s_calibrate, m, n_episodes, rng), 1 - alpha)
        power.append(float(np.mean(episode_stat(s_test_sub, m, n_episodes, rng) > thr)))
        fpr_r.append(float(np.mean(episode_stat(s_test_ref, m, n_episodes, rng) > thr)))
        fpr_n.append(float(np.mean(episode_stat(s_null, m, n_episodes, rng) > thr)))
        boots = []
        for _ in range(n_boot):
            c = rng.choice(s_calibrate, size=len(s_calibrate), replace=True)
            t = rng.choice(s_test_sub, size=len(s_test_sub), replace=True)
            thr_b = np.quantile(episode_stat(c, m, n_episodes // 4, rng), 1 - alpha)
            boots.append(np.mean(episode_stat(t, m, n_episodes // 4, rng) > thr_b))
        lo.append(float(np.percentile(boots, 2.5)))
        hi.append(float(np.percentile(boots, 97.5)))
        bl.append(int(m))
    return PowerCurve(
        budgets=bl,
        power=power,
        power_lo=lo,
        power_hi=hi,
        fpr_ref=fpr_r,
        fpr_null=fpr_n,
        alpha=alpha,
        n_episodes=n_episodes,
        queries_to_detect_80=_first_budget(bl, power, 0.80),
        queries_to_detect_95=_first_budget(bl, power, 0.95),
    )


def split_indices(
    n: int, fractions: tuple[float, ...], rng: np.random.Generator
) -> list[np.ndarray]:
    """Random disjoint index splits of range(n) with the given fractions (last gets the remainder)."""
    perm = rng.permutation(n)
    cuts = np.cumsum([round(f * n) for f in fractions[:-1]])
    return list(np.split(perm, cuts))
