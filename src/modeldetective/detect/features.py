"""Visible-string features for one probe response (IRIS style, arXiv 2607.20860 App. S23).

Two column sets matter:
- FULL: everything, including length and format features.
- CONTENT_ONLY: the length-ablated set. Every de-confounded claim must use this one, because
  a length or format difference can be a serving artifact rather than a model difference.

The content-only set here follows scripts/quant_twin.py --ablate-length in IRIS-audit
(drop lengths, counts, format flags, run lengths, compression, chi2; keep frequencies,
entropies, alternation, autocorrelation, transitions, positional balance).
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from itertools import pairwise

from modeldetective.probes.iris import Parsed, Probe

LENGTH_FORMAT_FEATURES = frozenset(
    {
        "response_chars",
        "response_non_ws_chars",
        "contains_newline",
        "contains_space",
        "contains_alpha_text",
        "invalid_chars",
        "checked_chars",
        "invalid_rate",
        "parsed_len",
        "exact_len",
        "parse_success",
        "length_error",
        "abs_length_error",
        "rel_length_error",
        "run_count",
        "mean_run",
        "std_run",
        "longest_run",
        "longest_run_ratio",
        "compression_ratio",
        "chi2_target",
        "number_num_digits",
        "alphabet_size",
        "expected_len",
    }
)


def _entropy(counts: Counter[str], n: int) -> float:
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)


def _ngram_entropy(s: str, k: int) -> float:
    if len(s) < k:
        return 0.0
    grams = Counter(s[i : i + k] for i in range(len(s) - k + 1))
    return _entropy(grams, len(s) - k + 1)


def _runs(s: str) -> list[int]:
    if not s:
        return []
    runs, cur = [], 1
    for a, b in pairwise(s):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs


def _autocorr(codes: list[int], lag: int) -> float:
    n = len(codes)
    if n <= lag + 1:
        return 0.0
    mean = sum(codes) / n
    var = sum((c - mean) ** 2 for c in codes)
    if var == 0:
        return 0.0
    cov = sum((codes[i] - mean) * (codes[i + lag] - mean) for i in range(n - lag))
    return cov / var


def extract_features(text: str, parsed: Parsed, probe: Probe) -> dict[str, float]:
    s = parsed.symbols
    n = len(s)
    k = probe.n_symbols
    counts = Counter(s)
    target = 1.0 / k
    f: dict[str, float] = {}

    # Format and length
    f["alphabet_size"] = k
    f["expected_len"] = probe.length
    f["response_chars"] = len(text)
    f["response_non_ws_chars"] = sum(1 for c in text if not c.isspace())
    f["contains_newline"] = float("\n" in text)
    f["contains_space"] = float(" " in text.strip())
    f["contains_alpha_text"] = float(any(c.isalpha() and c not in probe.alphabet for c in text))
    f["invalid_chars"] = parsed.invalid_chars
    f["checked_chars"] = parsed.checked_chars
    f["invalid_rate"] = parsed.invalid_rate
    f["parsed_len"] = n
    f["exact_len"] = float(n == probe.length)
    f["parse_success"] = float(parsed.parse_success)
    f["length_error"] = n - probe.length
    f["abs_length_error"] = abs(n - probe.length)
    f["rel_length_error"] = abs(n - probe.length) / probe.length

    # Symbol distribution
    freqs = {sym: counts.get(sym, 0) / n if n else 0.0 for sym in probe.alphabet}
    f["chi2_target"] = (
        sum((counts.get(sym, 0) - n * target) ** 2 / (n * target) for sym in probe.alphabet)
        if n
        else 0.0
    )
    f["entropy"] = _entropy(counts, n)
    f["entropy_norm"] = f["entropy"] / math.log2(k) if k > 1 else 0.0
    f["l1_target"] = sum(abs(p - target) for p in freqs.values())
    f["max_freq"] = max(freqs.values()) if n else 0.0
    f["min_freq"] = min(freqs.values()) if n else 0.0
    f["freq_range"] = f["max_freq"] - f["min_freq"]
    f["mode_count_ratio"] = (max(counts.values()) / n) if n else 0.0

    # Runs and sequence structure
    runs = _runs(s)
    f["run_count"] = len(runs)
    f["mean_run"] = (sum(runs) / len(runs)) if runs else 0.0
    f["std_run"] = (
        math.sqrt(sum((r - f["mean_run"]) ** 2 for r in runs) / len(runs)) if runs else 0.0
    )
    f["longest_run"] = max(runs) if runs else 0
    f["longest_run_ratio"] = (max(runs) / n) if n else 0.0
    f["alternation_rate"] = sum(1 for a, b in pairwise(s) if a != b) / (n - 1) if n > 1 else 0.0
    f["expected_alternation"] = 1.0 - target
    f["bigram_entropy"] = _ngram_entropy(s, 2)
    f["trigram_entropy"] = _ngram_entropy(s, 3)
    f["compression_ratio"] = (len(zlib.compress(s.encode())) / n) if n else 0.0
    codes = [probe.alphabet.index(c) for c in s]
    for lag in (1, 2, 3):
        f[f"autocorr_lag{lag}"] = _autocorr(codes, lag)
    f["first_symbol_code"] = codes[0] if codes else -1
    f["last_symbol_code"] = codes[-1] if codes else -1

    # Per-symbol counts, frequencies, transitions (alphabets up to 10 symbols)
    if k <= 10:
        for sym in probe.alphabet:
            f[f"count_{sym}"] = counts.get(sym, 0)
            f[f"freq_{sym}"] = freqs[sym]
        trans = Counter(pairwise(s))
        for a in probe.alphabet:
            for b in probe.alphabet:
                f[f"trans_{a}_{b}"] = trans.get((a, b), 0) / (n - 1) if n > 1 else 0.0

    # Positional balance (binary alphabets)
    if k == 2:
        sym0 = probe.alphabet[0]
        half = n // 2
        f["alt0_mismatch"] = (
            sum(1 for i, c in enumerate(s) if c != probe.alphabet[i % 2]) / n if n else 0.0
        )
        f["alt1_mismatch"] = (
            sum(1 for i, c in enumerate(s) if c != probe.alphabet[(i + 1) % 2]) / n if n else 0.0
        )
        f["first_half_symbol0_freq"] = (s[:half].count(sym0) / half) if half else 0.0
        f["second_half_symbol0_freq"] = (s[half:].count(sym0) / (n - half)) if n - half else 0.0
        f["half_freq_gap"] = abs(f["first_half_symbol0_freq"] - f["second_half_symbol0_freq"])
        q = max(n // 4, 1)
        quarters = [s[i * q : (i + 1) * q].count(sym0) / q for i in range(4)] if n >= 4 else []
        qm = sum(quarters) / 4 if quarters else 0.0
        f["quarter_freq_std"] = (
            math.sqrt(sum((x - qm) ** 2 for x in quarters) / 4) if quarters else 0.0
        )

    # Number probes
    if probe.parser == "number":
        v = parsed.number_value
        f["number_found"] = float(v is not None)
        f["number_value"] = v if v is not None else -1
        f["number_log10"] = math.log10(abs(v)) if v else 0.0
        f["number_num_digits"] = len(str(abs(v))) if v is not None else 0
        f["number_parity"] = (v % 2) if v is not None else -1
        f["number_mod10"] = (v % 10) if v is not None else -1
        f["number_mod100"] = (v % 100) if v is not None else -1

    return f


def select_columns(columns: list[str], content_only: bool) -> list[str]:
    """Return the feature columns to use. Content-only drops length/format and count_* columns."""
    if not content_only:
        return list(columns)
    return [c for c in columns if c not in LENGTH_FORMAT_FEATURES and not c.startswith("count_")]
