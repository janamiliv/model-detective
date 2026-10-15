"""Fixed held-out text for the KL-divergence measurement: configs/kl_text.txt.

Source: WikiText-2 raw, test split (Salesforce/wikitext, wikitext-2-raw-v1), the standard
corpus for llama.cpp perplexity and KL measurements, CC BY-SA 3.0. We keep the first
``MAX_CHARS`` characters so the file is small and identical for every variant. The probes never
contain this text, so there is no overlap with the detection prompts.

    uv run python -m modeldetective.harm.kl_text
"""

from __future__ import annotations

from pathlib import Path

OUT = Path("configs/kl_text.txt")
MAX_CHARS = 160_000  # about 40k tokens: 64 chunks of 512 tokens with margin


REPO = "Salesforce/wikitext"
FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def build(out: Path = OUT, max_chars: int = MAX_CHARS) -> Path:
    import json

    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    rev = HfApi().dataset_info(REPO).sha
    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=rev)
    text = "".join(pd.read_parquet(path)["text"])[:max_chars]
    out.write_text(text, encoding="utf-8", newline="\n")
    meta = {"repo": REPO, "file": FILE, "revision": rev, "max_chars": max_chars}
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta) + "\n", encoding="utf-8", newline="\n"
    )
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}: {p.stat().st_size} bytes")
