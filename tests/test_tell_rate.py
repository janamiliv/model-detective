"""Tell-rate detector on synthetic distributions with known answers."""

import numpy as np
import pytest

from modeldetective.detect.features import extract_features, select_columns
from modeldetective.detect.tell_rate import tell_rate
from modeldetective.probes.iris import IRIS_PROBES, parse_response

BITS16 = IRIS_PROBES["short_bits_16"]


def _bits_pool(rng: np.random.Generator, n: int, p_one: float) -> np.ndarray:
    rows = []
    for _ in range(n):
        s = "".join("1" if rng.random() < p_one else "0" for _ in range(16))
        f = extract_features(s, parse_response(s, BITS16), BITS16)
        rows.append(f)
    cols = select_columns(list(rows[0]), content_only=True)
    return np.array([[r[c] for c in cols] for r in rows])


@pytest.mark.parametrize("seed", [0, 1])
def test_null_stays_near_alpha(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a, b = _bits_pool(rng, 150, 0.5), _bits_pool(rng, 150, 0.5)
    r = tell_rate(a, b, alpha=0.05, n_splits=10, n_estimators=100, seed=seed)
    assert abs(r.ref_false_alarm_mean - 0.05) < 0.03
    assert r.q_mean < 0.15  # same distribution: no reliable tells
    assert not r.clears_floor
    assert 0.35 < r.auroc_mean < 0.65


def test_strong_shift_is_detected() -> None:
    rng = np.random.default_rng(3)
    a, b = _bits_pool(rng, 150, 0.5), _bits_pool(rng, 150, 0.85)
    r = tell_rate(a, b, alpha=0.05, n_splits=10, n_estimators=100, seed=3)
    assert r.q_mean > 0.6
    assert r.clears_floor
    assert r.auroc_mean > 0.9
    assert r.budget(eps=0.1) < 60


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        tell_rate(np.zeros((5, 3)), np.zeros((5, 4)))
