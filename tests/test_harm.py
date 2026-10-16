"""Harm metrics on synthetic answer vectors; prompt formatting; KL log parsing."""

import numpy as np

from modeldetective.harm.eval import parse_letter
from modeldetective.harm.kl import parse_kl_output
from modeldetective.harm.metrics import harm
from modeldetective.harm.mmlu import format_prompt


def test_harm_flips_and_accuracy() -> None:
    ref = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    var = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    r = harm(ref, var, var_invalid=1, n_boot=200)
    assert r.n == 10
    assert r.accuracy == 0.5 and r.accuracy_drop == 0.1
    assert r.flip_rate == 0.3
    assert r.flips_correct_to_wrong == 2 and r.flips_wrong_to_correct == 1
    assert r.invalid_answers == 1
    assert r.flip_rate_ci[0] <= r.flip_rate <= r.flip_rate_ci[1]


def test_harm_identical_runs_have_zero_flips() -> None:
    ref = np.array([1, 0, 1, 1])
    r = harm(ref, ref.copy(), n_boot=50)
    assert r.flip_rate == 0.0 and r.flip_rate_ci == (0.0, 0.0) and r.accuracy_drop == 0.0


def test_format_prompt_and_parse_letter() -> None:
    q = {
        "subject": "high_school_biology",
        "question": "What is X?",
        "choices": ["a", "b", "c", "d"],
    }
    p = format_prompt(q)
    assert "high school biology" in p and p.endswith("Answer:") and "D. d" in p
    assert parse_letter(" B") == "B"
    assert parse_letter("(c)") == "C"
    assert parse_letter("The correct answer is B.") == "B"
    assert parse_letter("The correct answer is **C**") == "C"
    assert parse_letter("Answer: (D)") == "D"
    assert parse_letter("A cat is an animal") is None  # word starting with A is not an answer
    assert parse_letter("The question is asking about") is None
    assert parse_letter("") is None


def test_parse_kl_output() -> None:
    log = """
    ====== KL divergence statistics ======
    Mean    KLD:   0.012345 ±   0.000123
    Maximum KLD:   1.234567
    99.9%   KLD:   0.456789
    99.0%   KLD:   0.234567
    Median  KLD:   0.001234
    ====== Token probability statistics ======
    Mean    Δp: -0.123 ± 0.010 %
    Same top p: 95.123 ± 0.200 %
    """
    r = parse_kl_output(log)
    assert r["mean_kl"] == 0.012345 and r["mean_kl_se"] == 0.000123
    assert r["max_kl"] == 1.234567 and r["median_kl"] == 0.001234
    assert r["top1_agreement"] == 95.123


def test_parse_kl_output_negative_rounding() -> None:
    log = "Mean    KLD:  -0.000015 ±   0.000000\nMedian  KLD:  -0.000014\n99.0%   KLD:   0.000000\n"
    r = parse_kl_output(log)
    assert r["mean_kl"] == -0.000015 and r["median_kl"] == -0.000014 and r["kl_99pct"] == 0.0
