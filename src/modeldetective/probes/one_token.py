"""One Token Is Enough battery (Bruckner 2026, arXiv 2607.10252) and answer normalization.

Prompts and system prompts are read verbatim from configs/one_token_prompts.zenodo.json (the
author's PAMELA harness, MIT). Only the paper-1 fingerprint tasks are used (10 tasks x 4
languages = 40 cells); the paper-2 coordination tasks are excluded, as in the paper.

Normalization follows Sec. IV-B of the paper: NFC, strip punctuation and quotes, case fold,
map Arabic-Indic and (simple) Chinese numerals to digits, take the first whitespace token.
Deviation: no per-language color lexicon; color answers are compared as normalized strings.
Each answer gets a class: valid / invalid / refusal / empty. Nothing is dropped silently.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from modeldetective.probes.iris import Probe

BATTERY_FILE = Path("configs/one_token_prompts.zenodo.json")
LANGS = ["en", "ru", "zh", "ar"]
MAX_TOKENS = 16  # paper: max_tokens 16 (pre-registration said 12; raised for OpenAI endpoints)

_ARABIC_INDIC = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_ARABIC_INDIC.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
_FULLWIDTH = {ord(c): str(i) for i, c in enumerate("０１２３４５６７８９")}
_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_REFUSAL = re.compile(
    r"(i can'?t|i cannot|i'm unable|as an ai|i am unable|не могу|无法|不能|لا أستطيع|لا يمكنني)",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[\"'«»“”‘’`.,;:!?()\[\]{}<>*_~\-—–/\\|]+")
_INT = re.compile(r"^-?\d+$")


@dataclass(frozen=True)
class OneTokenTask:
    id: str
    answer_space: str
    normalize_as: str
    prompts: dict[str, str]


def load_battery(path: Path = BATTERY_FILE) -> tuple[dict[str, str], list[OneTokenTask]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    tasks = [
        OneTokenTask(t["id"], t["answer_space"], t.get("normalize_as", ""), t["prompts"])
        for t in d["tasks"]
        if int(t.get("paper", 1)) == 1
    ]
    return d["system_prompts"], tasks


def battery_probes(path: Path = BATTERY_FILE) -> list[tuple[Probe, str]]:
    """(Probe, system_prompt) per (task, language). Probe ids are 'ot__<task>__<lang>'."""
    system, tasks = load_battery(path)
    out = []
    for t in tasks:
        for lang in LANGS:
            out.append(
                (
                    Probe(f"ot__{t.id}__{lang}", t.prompts[lang], "", 1, "raw", MAX_TOKENS),
                    system[lang],
                )
            )
    return out


def _cn_number(s: str) -> str | None:
    """Very small Chinese numeral mapper for 0-100 (十, 二十三, 一百)."""
    if not s or any(c not in _CN_DIGITS and c not in "十百" for c in s):
        return None
    if s == "十":
        return "10"
    if s == "一百" or s == "百":
        return "100"
    total = 0
    if "十" in s:
        tens, _, ones = s.partition("十")
        total = (_CN_DIGITS[tens] if tens else 1) * 10 + (_CN_DIGITS[ones] if ones else 0)
    else:
        total = _CN_DIGITS.get(s, -1)
        if total < 0:
            return None
    return str(total)


def normalize_answer(text: str | None, task: OneTokenTask) -> tuple[str, str]:
    """Return (normalized_answer, class) with class in valid / invalid / refusal / empty."""
    if text is None or not text.strip():
        return "", "empty"
    t = unicodedata.normalize("NFC", text)
    if _REFUSAL.search(t):
        return "", "refusal"
    t = t.translate(_ARABIC_INDIC).translate(_FULLWIDTH)
    t = _PUNCT.sub(" ", t).strip().casefold()
    if not t:
        return "", "empty"
    first = t.split()[0]
    cn = _cn_number(first)
    if cn is not None:
        first = cn
    if task.answer_space.startswith("integer"):
        if not _INT.match(first):
            return first, "invalid"
        lo, hi = (int(x) for x in task.answer_space.split()[1].split("-"))
        return first, "valid" if lo <= int(first) <= hi else "invalid"
    if task.answer_space.startswith("heads"):
        pairs = {"heads", "tails", "орёл", "орел", "решка", "正面", "反面", "صورة", "كتابة"}
        return first, "valid" if first in pairs else "invalid"
    return first, "valid"
