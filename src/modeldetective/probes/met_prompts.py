"""Build the MET-style natural prompt set (configs/met_prompts.yaml) reproducibly.

Model Equality Testing (Gao, Liang, Guestrin, ICLR 2025) used 25 random 100-character Wikipedia
snippets from en/de/es/fr/ru with the template below. We fix 25 article titles (5 per language),
take the first 100 characters of each lead paragraph via the Wikipedia REST summary API, and
commit the result so the prompt set is a frozen artifact. Text is CC BY-SA 4.0 (Wikipedia);
the YAML carries the attribution.

    uv run python -m modeldetective.probes.met_prompts
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

OUT = Path("configs/met_prompts.yaml")
TEMPLATE = (
    "Continue the paragraph. Do not output anything except the continuation to the paragraph. "
    'Start the continuation immediately. "{snippet}..."'
)
# Deterministic, ordinary encyclopedia topics; five per language, chosen once on 2026-10-15.
TITLES: dict[str, list[str]] = {
    "en": ["Photosynthesis", "Danube", "Chess", "Lighthouse", "Honey"],
    "de": ["Schwarzwald", "Buchdruck", "Nordsee", "Fahrrad", "Bienenwachs"],
    "es": ["Guadalquivir", "Flamenco", "Mate_(infusión)", "Volcán", "Alfarería"],
    "fr": ["Loire", "Boulangerie", "Mont_Blanc", "Vitrail", "Camembert_(fromage)"],
    "ru": ["Байкал", "Самовар", "Балалайка", "Берёза", "Шахматы"],
}
UA = "model-detective/0.1 (https://github.com/janamiliv/model-detective; research) httpx/0.28"


def fetch_snippet(lang: str, title: str, n_chars: int = 100) -> str:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }
    r = httpx.get(url, params=params, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page:
        raise RuntimeError(f"no extract for {lang}:{title}: {page}")
    extract = " ".join(page["extract"].split())
    if len(extract) < n_chars:
        raise RuntimeError(f"extract too short for {lang}:{title} (disambiguation page?)")
    return extract[:n_chars]


def main() -> None:
    prompts = []
    i = 0
    for lang, titles in TITLES.items():
        for title in titles:
            snippet = fetch_snippet(lang, title)
            prompts.append(
                {
                    "id": f"met_{i:02d}_{lang}",
                    "lang": lang,
                    "source": f"https://{lang}.wikipedia.org/wiki/{title}",
                    "snippet": snippet,
                    "prompt": TEMPLATE.format(snippet=snippet),
                }
            )
            i += 1
    doc = {
        "description": "MET-style natural continuation prompts: 25 Wikipedia lead-paragraph snippets "
        "(100 chars), 5 per language (en/de/es/fr/ru), template from Gao et al. ICLR 2025 Box 1.",
        "license": "Snippets from Wikipedia, CC BY-SA 4.0; see each source URL.",
        "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "max_tokens": 50,
        "prompts": prompts,
    }
    OUT.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT} with {len(prompts)} prompts")


if __name__ == "__main__":
    main()
