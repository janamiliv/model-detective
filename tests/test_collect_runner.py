"""Collection runner invariants that need no server."""

from modeldetective.collect.runner import Cell, seed_for
from modeldetective.probes.iris import IRIS_PROBES


def test_seed_is_deterministic_and_unique() -> None:
    a = Cell("m", "f16", 1, IRIS_PROBES["short_bits_16"], 10)
    b = Cell("m", "f16", 2, IRIS_PROBES["short_bits_16"], 10)
    c = Cell("m", "f16", 1, IRIS_PROBES["single_number_100"], 10)
    assert seed_for(a, 0) == seed_for(a, 0)
    seeds = {seed_for(cell, i) for cell in (a, b, c) for i in range(10)}
    assert len(seeds) == 30  # distinct across runs, probes and indices


def test_cell_label_is_filesystem_safe() -> None:
    cell = Cell("qwen3-0.6b", "Q4_K_M", 1, IRIS_PROBES["short_bits_16"], 5)
    assert cell.label == "qwen3-0.6b__Q4_K_M__run1__short_bits_16"
    assert "/" not in cell.label and " " not in cell.label
