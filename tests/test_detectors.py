"""JSD, Hamming MMD and the power procedure on synthetic data with known answers."""

import math

import numpy as np
import pytest

from modeldetective.detect.jsd import battery_jsd, jsd
from modeldetective.detect.mmd import encode, hamming_kernel, mmd2, permutation_pvalue
from modeldetective.detect.power import episode_means, power_curve_from_scores, split_indices


def test_jsd_bounds_and_symmetry() -> None:
    assert jsd(["a"] * 10, ["a"] * 10) == 0.0
    assert math.isclose(jsd(["a"] * 10, ["b"] * 10), 1.0)
    p, q = ["a", "a", "b"], ["a", "b", "b"]
    assert math.isclose(jsd(p, q), jsd(q, p))
    assert 0.0 < jsd(p, q) < 1.0
    assert math.isnan(jsd([], ["a"]))


def test_battery_jsd_respects_min_n() -> None:
    a = {"c1": ["x"] * 20, "c2": ["x"] * 5, "c3": ["y"] * 20}
    b = {"c1": ["x"] * 20, "c2": ["y"] * 5, "c3": ["x"] * 20}
    mean, n = battery_jsd(a, b, min_n=10)
    assert n == 2 and math.isclose(mean, 0.5)  # c2 excluded; c1 = 0, c3 = 1


def test_hamming_kernel_and_padding() -> None:
    a = encode(["abc", "ab"], 4)
    k = hamming_kernel(a, a)
    assert math.isclose(k[0, 0], 1.0)
    # "abc_" vs "ab__": positions 0,1 agree, position 3 pad==pad agrees -> 3/4
    assert math.isclose(k[0, 1], 0.75)


def test_mmd_null_and_shift() -> None:
    rng = np.random.default_rng(0)

    def pool(p1: float, n: int) -> list[str]:
        return ["".join("1" if rng.random() < p1 else "0" for _ in range(16)) for _ in range(n)]

    a, b = encode(pool(0.5, 60), 16), encode(pool(0.5, 60), 16)
    c = encode(pool(0.9, 60), 16)
    p_null, _ = permutation_pvalue(a, b, n_perm=200, seed=1)
    p_alt, stat_alt = permutation_pvalue(a, c, n_perm=200, seed=1)
    assert p_null > 0.05
    assert p_alt < 0.01 and stat_alt > 0
    assert abs(mmd2(a, b)) < 0.02


def test_mmd_prompt_restriction() -> None:
    a = encode(["aaaa"] * 3 + ["bbbb"] * 3, 4)
    b = encode(["aaaa"] * 3 + ["bbbb"] * 3, 4)
    pa = np.array([0, 0, 0, 1, 1, 1])
    # Identical per-prompt distributions: restricted to same-prompt pairs every kernel value is 1,
    # so MMD is exactly 0. Unrestricted, cross-prompt pairs (k = 0) enter the within terms only
    # through the diagonal exclusion asymmetry and the statistic is no longer 0.
    assert math.isclose(mmd2(a, b, pa, pa), 0.0)
    assert not math.isclose(mmd2(a, b), 0.0)


def test_power_procedure_recovers_known_shift() -> None:
    rng = np.random.default_rng(5)
    ref = rng.normal(0, 1, 2000)
    cal, test_ref, null = ref[:1000], ref[1000:1500], rng.normal(0, 1, 500)
    sub = rng.normal(0.5, 1, 1000)  # small per-query shift: needs several queries to detect
    pc = power_curve_from_scores(cal, test_ref, null, sub, n_episodes=400, n_boot=30, seed=1)
    assert pc.power[0] < 0.5  # one query is not enough at d = 0.5
    assert pc.power[-1] > 0.95  # 256 queries is plenty
    assert pc.queries_to_detect_80 is not None and 4 <= pc.queries_to_detect_80 <= 64
    assert all(abs(f - 0.05) < 0.05 for f in pc.fpr_null)  # null stays near alpha at every m
    assert all(lo <= p <= hi for lo, p, hi in zip(pc.power_lo, pc.power, pc.power_hi, strict=True))


def test_power_no_detection_reported_as_none() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 800)
    pc = power_curve_from_scores(
        ref[:400],
        ref[400:600],
        rng.normal(0, 1, 200),
        rng.normal(0, 1, 400),
        n_episodes=200,
        n_boot=10,
        seed=3,
    )
    assert pc.queries_to_detect_80 is None
    assert max(pc.power) < 0.3


def test_episode_means_and_splits() -> None:
    rng = np.random.default_rng(0)
    s = np.arange(10, dtype=float)
    e = episode_means(s, 4, 100, rng)
    assert e.shape == (100,) and e.min() >= 1.5 and e.max() <= 7.5
    parts = split_indices(100, (0.5, 0.25, 0.25), rng)
    assert [len(p) for p in parts] == [50, 25, 25]
    assert len(set(np.concatenate(parts))) == 100


@pytest.mark.parametrize("length", [1, 8])
def test_encode_truncates(length: int) -> None:
    assert encode(["abcdefgh"], length).shape == (1, length)
