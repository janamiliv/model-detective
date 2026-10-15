"""Jensen-Shannon divergence between categorical answer distributions (One Token style).

JSD is base 2, bounded in [0, 1], symmetric, defined on disjoint supports, not square-rooted,
exactly as in Bruckner (2026) Eq. 1. The battery distance is the mean over cells where both
sides have at least ``min_n`` valid answers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np


def _dist(answers: Iterable[str]) -> Counter[str]:
    return Counter(answers)


def jsd(p: Counter[str] | Sequence[str], q: Counter[str] | Sequence[str]) -> float:
    cp = p if isinstance(p, Counter) else _dist(p)
    cq = q if isinstance(q, Counter) else _dist(q)
    np_, nq = sum(cp.values()), sum(cq.values())
    if np_ == 0 or nq == 0:
        return float("nan")
    support = set(cp) | set(cq)
    pv = np.array([cp.get(k, 0) / np_ for k in support])
    qv = np.array([cq.get(k, 0) / nq for k in support])
    m = 0.5 * (pv + qv)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)


def battery_jsd(
    cells_a: dict[str, Sequence[str]], cells_b: dict[str, Sequence[str]], min_n: int = 10
) -> tuple[float, int]:
    """Mean JSD over shared cells with >= min_n valid answers on both sides. Returns (mean, n_cells)."""
    vals = []
    for cell in cells_a.keys() & cells_b.keys():
        a, b = cells_a[cell], cells_b[cell]
        if len(a) >= min_n and len(b) >= min_n:
            vals.append(jsd(a, b))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)
