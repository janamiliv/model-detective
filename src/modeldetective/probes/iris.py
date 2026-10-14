"""IRIS-style random-generation probes and the response parser.

Prompts are verbatim from Zhang, Zhang, Qin (2026), arXiv 2607.20860, App. S22 Tables 19-20,
and from scripts/m_scaling_experiment.py in github.com/Photen/IRIS-audit (commit 3d859ab).
Parser semantics follow rf_preexperiment.py::parse_response in that repo:
- "char" parser keeps every character of the response that is in the alphabet, in order;
- "number" parser (IRIS) takes the first ``-?\\d+`` match;
- "number_last" parser (ours, prereg section 7) takes the LAST integer and flags prompt echo,
  because small models echo "between 1 and 100" and the first integer is then the prompt's "1".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SEPARATORS = set(" \t\r\n,.;:|/\\-_'\"`[]{}()<>")
_NUMBER_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Probe:
    id: str
    prompt: str
    alphabet: str  # ordered symbols; for "number" probes this is the digit alphabet
    length: int  # requested length (1 for single draws)
    parser: str  # "char" or "number"
    max_tokens: int  # IRIS decode cap: 48 for single draws, max(72, 3L+40) otherwise

    @property
    def n_symbols(self) -> int:
        return len(self.alphabet)


def iris_max_tokens(length: int) -> int:
    return 48 if length == 1 else max(72, 3 * length + 40)


def _p(id_: str, prompt: str, alphabet: str, length: int, parser: str = "char") -> Probe:
    return Probe(id_, prompt, alphabet, length, parser, iris_max_tokens(length))


_BITS = (
    "Generate exactly {L} symbols by independently choosing uniformly between 0 and 1. "
    "Output the sequence only, with no spaces and no explanation."
)
_DIGITS = (
    "Generate exactly {L} independent uniformly random digits from 0 to 9. "
    "Output only the compact digit string, no separators."
)

IRIS_PROBES: dict[str, Probe] = {
    p.id: p
    for p in [
        _p(
            "single_digit",
            "Output one uniformly random digit from 0 to 9. Output only the single digit, nothing else.",
            "0123456789",
            1,
        ),
        _p(
            "single_number_100",
            "Choose one uniformly random integer between 1 and 100 inclusive. Output only the integer.",
            "0123456789",
            1,
            parser="number",
        ),
        _p(
            "single_coin",
            "Flip one fair coin. Output only a single character: H for heads or T for tails.",
            "HT",
            1,
        ),
        _p(
            "single_number_100_last",
            "Choose one uniformly random integer between 1 and 100 inclusive. Output only the integer.",
            "0123456789",
            1,
            parser="number_last",
        ),
        _p("short_bits_8", _BITS.format(L=8), "01", 8),
        _p("short_bits_16", _BITS.format(L=16), "01", 16),
        _p("short_bits_32", _BITS.format(L=32), "01", 32),
        _p("short_digits_8", _DIGITS.format(L=8), "0123456789", 8),
        _p("short_digits_16", _DIGITS.format(L=16), "0123456789", 16),
    ]
}


@dataclass(frozen=True)
class Parsed:
    symbols: str  # parsed symbol sequence (may be empty)
    invalid_chars: int
    checked_chars: int
    number_value: int | None  # only for number probes
    echo: bool = False  # number_last only: every integer in the response also occurs in the prompt

    @property
    def invalid_rate(self) -> float:
        return self.invalid_chars / self.checked_chars if self.checked_chars else 0.0

    @property
    def parse_success(self) -> bool:
        return len(self.symbols) > 0 and self.invalid_rate <= 0.25


def parse_response(text: str, probe: Probe) -> Parsed:
    """Never raises and never drops the row: an unparseable response gives empty symbols."""
    alphabet = set(probe.alphabet)
    checked = [c for c in text if c not in SEPARATORS]
    invalid = sum(1 for c in checked if c not in alphabet)
    if probe.parser == "number":
        m = _NUMBER_RE.search(text)
        if m is None:
            return Parsed("", invalid, len(checked), None)
        digits = m.group(0).lstrip("-")
        return Parsed(digits, invalid, len(checked), int(m.group(0)))
    if probe.parser == "raw":
        # One Token battery: keep the response; normalization happens in probes/one_token.py.
        t = text.strip()
        return Parsed(t, invalid, len(checked), None)
    if probe.parser == "number_last":
        found = _NUMBER_RE.findall(text)
        if not found:
            return Parsed("", invalid, len(checked), None)
        prompt_ints = set(_NUMBER_RE.findall(probe.prompt))
        echo = all(x in prompt_ints for x in found)
        # Prefer the last integer that is not a prompt echo; fall back to the last integer.
        non_echo = [x for x in found if x not in prompt_ints]
        pick = (non_echo or found)[-1]
        return Parsed(pick.lstrip("-"), invalid, len(checked), int(pick), echo=echo)
    symbols = "".join(c for c in text if c in alphabet)
    return Parsed(symbols, invalid, len(checked), None)
