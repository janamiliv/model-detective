"""Parser and feature tests on hand-checked strings."""

import math

from modeldetective.detect.features import extract_features, select_columns
from modeldetective.probes.iris import IRIS_PROBES, iris_max_tokens, parse_response

BITS16 = IRIS_PROBES["short_bits_16"]
NUM100 = IRIS_PROBES["single_number_100"]


def test_decode_caps_match_iris() -> None:
    assert iris_max_tokens(1) == 48
    assert iris_max_tokens(8) == 72
    assert iris_max_tokens(16) == 88
    assert iris_max_tokens(32) == 136
    assert BITS16.max_tokens == 88 and NUM100.max_tokens == 48


def test_char_parser_keeps_alphabet_chars_in_order() -> None:
    p = parse_response("Sure! 0101 1100\n", BITS16)
    assert p.symbols == "01011100"
    assert p.number_value is None
    # separators are not counted as invalid; letters are
    assert p.invalid_chars == 5  # "Sure!" (the exclamation mark is not a separator)
    assert not p.parse_success  # invalid_rate = 4/13 > 0.25


def test_char_parser_clean_string() -> None:
    p = parse_response("0110100110010110", BITS16)
    assert p.symbols == "0110100110010110"
    assert p.invalid_chars == 0 and p.parse_success


def test_number_parser_first_integer() -> None:
    p = parse_response("I pick 42.", NUM100)
    assert p.number_value == 42 and p.symbols == "42"
    assert parse_response("no digits here", NUM100).number_value is None
    assert parse_response("", NUM100).symbols == ""


def test_features_on_known_string() -> None:
    text = "0101010101010101"
    f = extract_features(text, parse_response(text, BITS16), BITS16)
    assert f["parsed_len"] == 16 and f["exact_len"] == 1.0
    assert math.isclose(f["entropy"], 1.0)
    assert math.isclose(f["alternation_rate"], 1.0)
    assert f["longest_run"] == 1 and f["run_count"] == 16
    assert math.isclose(f["freq_0"], 0.5) and f["count_1"] == 8
    assert math.isclose(f["trans_0_1"], 8 / 15)
    assert math.isclose(f["alt0_mismatch"], 0.0)
    assert f["autocorr_lag1"] < -0.9


def test_features_on_empty_response_do_not_crash() -> None:
    f = extract_features("", parse_response("", BITS16), BITS16)
    assert f["parsed_len"] == 0 and f["entropy"] == 0.0 and f["parse_success"] == 0.0


def test_content_only_drops_length_and_counts() -> None:
    text = "0110100110010110"
    f = extract_features(text, parse_response(text, BITS16), BITS16)
    cols = select_columns(list(f), content_only=True)
    assert "parsed_len" not in cols and "response_chars" not in cols
    assert "count_0" not in cols and "freq_0" in cols
    assert "entropy" in cols and "trans_0_1" in cols
    assert len(cols) < len(f)
