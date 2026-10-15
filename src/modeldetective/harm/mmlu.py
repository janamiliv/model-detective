"""Fixed MMLU subset for the harm measurement (prereg section 10).

1,000 questions drawn once with seed 20261014 from the MMLU test split (cais/mmlu, config "all"),
stratified by subject (proportional), and saved to configs/mmlu_subset.jsonl so every variant
sees exactly the same questions in the same order. Prompt format is the standard zero-shot
multiple-choice layout; the model answers with a single letter, read from the first generated
token (greedy decoding).

    uv run python -m modeldetective.harm.mmlu   # builds the subset file once
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

SUBSET = Path("configs/mmlu_subset.jsonl")
SEED = 20261014
N = 1000
LETTERS = "ABCD"


def format_prompt(q: dict) -> str:
    choices = "\n".join(f"{letter}. {c}" for letter, c in zip(LETTERS, q["choices"], strict=True))
    return (
        f"The following is a multiple choice question about {q['subject'].replace('_', ' ')}. "
        "Answer with the letter of the correct option only.\n\n"
        f"{q['question']}\n{choices}\nAnswer:"
    )


MMLU_REPO = "cais/mmlu"
MMLU_FILE = "all/test-00000-of-00001.parquet"


def load_mmlu_test(revision: str | None = None) -> tuple[pd.DataFrame, str]:
    """MMLU test split as a DataFrame via the hub client (the datasets library's own HTTP
    path fails on this machine's Windows certificate store). Returns (df, resolved revision)."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    rev = revision or HfApi().dataset_info(MMLU_REPO).sha
    path = hf_hub_download(MMLU_REPO, MMLU_FILE, repo_type="dataset", revision=rev)
    return pd.read_parquet(path), rev


def build_subset(out: Path = SUBSET, n: int = N, seed: int = SEED) -> Path:
    import numpy as np

    df, rev = load_mmlu_test()
    ds = df.to_dict("records")
    rng = np.random.default_rng(seed)
    by_subject: dict[str, list[int]] = {}
    for i, row in enumerate(ds):
        by_subject.setdefault(row["subject"], []).append(i)
    total = len(ds)
    picked: list[int] = []
    for _subject, idx in sorted(by_subject.items()):
        k = round(n * len(idx) / total)
        picked.extend(rng.choice(idx, size=min(k, len(idx)), replace=False).tolist())
    # Proportional rounding can land a few off n; trim or top up deterministically.
    picked = sorted(set(picked))
    if len(picked) > n:
        picked = rng.choice(picked, size=n, replace=False).tolist()
    while len(picked) < n:
        cand = int(rng.integers(total))
        if cand not in picked:
            picked.append(cand)
    picked = sorted(picked)
    rng.shuffle(picked)
    meta = {"repo": MMLU_REPO, "file": MMLU_FILE, "revision": rev, "seed": seed, "n": n}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for qid, i in enumerate(picked):
            row = ds[int(i)]
            f.write(
                json.dumps(
                    {
                        "qid": qid,
                        "source_index": int(i),
                        "subject": row["subject"],
                        "question": row["question"],
                        "choices": list(row["choices"]),
                        "answer": LETTERS[int(row["answer"])],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out


def load_subset(path: Path = SUBSET) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if "_meta" not in r]


if __name__ == "__main__":
    p = build_subset()
    print(f"wrote {p} with {len(load_subset(p))} questions")
