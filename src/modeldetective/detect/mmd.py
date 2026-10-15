"""Maximum Mean Discrepancy with the Hamming string kernel (Model Equality Testing).

Gao, Liang, Guestrin (ICLR 2025), Eq. 7: k(y, y') = number of positions at which two completions
agree after right-padding to length L. We follow the released package's normalisation (kernel
divided by L so k(y, y) = 1, diagonal excluded, cross-prompt pairs excluded) and compute the
unbiased statistic

    MMD^2 = mean k(x, x') [x != x' in A] + mean k(y, y') [y != y' in B] - 2 mean k(x, y).

Sequences are compared in Unicode code-point space (what an API shows). p-values come from a
permutation test on the pooled sample. The ``model-equality-testing`` package is used only as a
cross-check in tests, never as the implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PAD = -1


def encode(texts: Sequence[str], length: int) -> np.ndarray:
    """Code points, truncated or right-padded with PAD to ``length``. Shape (n, length)."""
    out = np.full((len(texts), length), PAD, dtype=np.int64)
    for i, t in enumerate(texts):
        cps = [ord(c) for c in t[:length]]
        out[i, : len(cps)] = cps
    return out


def hamming_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised Hamming agreement between every row of a and every row of b. Shape (na, nb)."""
    length = a.shape[1]
    return (a[:, None, :] == b[None, :, :]).sum(axis=2) / length


def mmd2(
    a: np.ndarray,
    b: np.ndarray,
    prompts_a: np.ndarray | None = None,
    prompts_b: np.ndarray | None = None,
) -> float:
    """Unbiased MMD^2 with the Hamming kernel. With prompt ids, only same-prompt pairs count."""
    kaa, kbb, kab = hamming_kernel(a, a), hamming_kernel(b, b), hamming_kernel(a, b)
    if prompts_a is not None and prompts_b is not None:
        maa = prompts_a[:, None] == prompts_a[None, :]
        mbb = prompts_b[:, None] == prompts_b[None, :]
        mab = prompts_a[:, None] == prompts_b[None, :]
    else:
        maa = np.ones_like(kaa, dtype=bool)
        mbb = np.ones_like(kbb, dtype=bool)
        mab = np.ones_like(kab, dtype=bool)
    np.fill_diagonal(maa, False)
    np.fill_diagonal(mbb, False)
    t_aa = kaa[maa].mean() if maa.any() else 0.0
    t_bb = kbb[mbb].mean() if mbb.any() else 0.0
    t_ab = kab[mab].mean() if mab.any() else 0.0
    return float(t_aa + t_bb - 2 * t_ab)


def permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 500,
    seed: int = 0,
    prompts_a: np.ndarray | None = None,
    prompts_b: np.ndarray | None = None,
) -> tuple[float, float]:
    """(p-value, observed statistic). p = fraction of permuted statistics >= observed."""
    rng = np.random.default_rng(seed)
    obs = mmd2(a, b, prompts_a, prompts_b)
    pooled = np.vstack([a, b])
    pp = None if prompts_a is None else np.concatenate([prompts_a, prompts_b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        pa, pb = pooled[idx[:na]], pooled[idx[na:]]
        qa = None if pp is None else pp[idx[:na]]
        qb = None if pp is None else pp[idx[na:]]
        if mmd2(pa, pb, qa, qb) >= obs:
            count += 1
    return count / n_perm, obs
