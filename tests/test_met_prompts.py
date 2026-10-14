"""The committed MET-style prompt set is a frozen artifact; check its shape."""

from pathlib import Path

import yaml

from modeldetective.probes.met_prompts import TEMPLATE


def test_met_prompt_set_shape() -> None:
    d = yaml.safe_load(Path("configs/met_prompts.yaml").read_text(encoding="utf-8"))
    prompts = d["prompts"]
    assert len(prompts) == 25
    assert sorted({p["lang"] for p in prompts}) == ["de", "en", "es", "fr", "ru"]
    assert all(len(p["snippet"]) == 100 for p in prompts)
    assert all(p["prompt"] == TEMPLATE.format(snippet=p["snippet"]) for p in prompts)
    assert len({p["id"] for p in prompts}) == 25
    assert d["max_tokens"] == 50
