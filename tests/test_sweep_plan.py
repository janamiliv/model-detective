"""Sweep planning from the committed configs; no server, no model files."""

from pathlib import Path

import yaml

from modeldetective.collect.runner import seed_for
from modeldetective.collect.sweep import iris_probe_specs, plan_cells, plan_pools

MODELS = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
PROBES = yaml.safe_load(Path("configs/probes.yaml").read_text(encoding="utf-8"))


def test_primary_probe_is_first_and_largest() -> None:
    specs = iris_probe_specs(PROBES)
    assert specs[0][0] == "short_bits_16"
    assert specs[0][1] == 2000
    assert all(n <= 2000 for _, n in specs[1:])


def test_pools_have_reference_null_and_every_variant() -> None:
    pools = plan_pools("qwen3-0.6b", MODELS)
    labels = [(p.variant, p.run_id) for p in pools]
    assert labels[:2] == [("f16", 1), ("f16", 2)]
    assert {v for v, _ in labels[2:]} == {"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"}
    assert pools[0].file == pools[1].file  # same reference file, different run ids


def test_seeds_disjoint_across_all_pools_and_cells() -> None:
    specs = iris_probe_specs(PROBES)
    seeds: set[int] = set()
    count = 0
    for pool in plan_pools("qwen3-0.6b", MODELS):
        for cell in plan_cells(pool, specs):
            for i in range(50):
                seeds.add(seed_for(cell, i))
                count += 1
    assert len(seeds) == count
