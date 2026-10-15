"""Cross-check our Hamming MMD against the released model-equality-testing package (Gao et al.).

Skipped when the package is not installed (it is the optional `baseline` extra).
"""

import numpy as np
import pytest

from modeldetective.detect.mmd import encode, mmd2

met = pytest.importorskip("model_equality_testing")


def _package_mmd(a: np.ndarray, b: np.ndarray, pa: np.ndarray, pb: np.ndarray, m: int) -> float:
    from model_equality_testing.distribution import CompletionSample
    from model_equality_testing.tests import mmd_hamming

    s1 = CompletionSample(prompts=pa, completions=a, m=m)
    s2 = CompletionSample(prompts=pb, completions=b, m=m)
    return float(mmd_hamming(s1, s2))


def test_matches_reference_package_on_random_strings() -> None:
    rng = np.random.default_rng(7)
    alphabet = "ab"
    m, n_per, length = 3, 20, 12

    def pool(p_a: float) -> tuple[np.ndarray, np.ndarray]:
        texts, prompts = [], []
        for pid in range(m):
            for _ in range(n_per):
                texts.append("".join(alphabet[rng.random() > p_a] for _ in range(length)))
                prompts.append(pid)
        return encode(texts, length), np.array(prompts)

    a, pa = pool(0.5)
    b, pb = pool(0.8)
    ours = mmd2(a, b, pa, pb)
    theirs = _package_mmd(a, b, pa, pb, m)
    assert ours == pytest.approx(theirs, abs=1e-6)
