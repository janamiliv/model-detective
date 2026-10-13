"""Minimal client for the OpenAI-compatible endpoints of llama-server.

Only what the probes need: one request, sampled, optional logprobs. No retries that could
silently change the sample distribution; failures raise and are counted by the caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class SamplerConfig:
    """Sampling parameters sent with every request. The project pins one of these per study."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables top-k in llama.cpp
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    max_tokens: int = 16

    def as_request(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Completion:
    text: str
    n_tokens: int
    finish_reason: str | None
    seed: int | None
    # Per generated token: {token, logprob, top_logprobs: [{token, logprob}, ...]}
    logprobs: list[dict[str, Any]] | None


class LlamaClient:
    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LlamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(
        self,
        prompt: str,
        sampler: SamplerConfig,
        seed: int | None = None,
        n_logprobs: int = 0,
    ) -> Completion:
        """Raw completion: the prompt is sent as-is, with no chat template."""
        body: dict[str, Any] = {"prompt": prompt, **sampler.as_request()}
        if seed is not None:
            body["seed"] = seed
        if n_logprobs:
            body["logprobs"] = n_logprobs
        r = self._http.post("/v1/completions", json=body)
        r.raise_for_status()
        return _parse_choice(r.json(), seed)

    def chat(
        self,
        messages: list[dict[str, str]],
        sampler: SamplerConfig,
        seed: int | None = None,
        n_logprobs: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> Completion:
        """Chat completion: the server applies the model's chat template."""
        body: dict[str, Any] = {"messages": messages, **sampler.as_request()}
        if seed is not None:
            body["seed"] = seed
        if n_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = n_logprobs
        if extra:
            body.update(extra)
        r = self._http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return _parse_choice(r.json(), seed)


def _parse_choice(data: dict[str, Any], seed: int | None) -> Completion:
    choice = data["choices"][0]
    text = (choice["message"].get("content") if "message" in choice else choice.get("text")) or ""
    lp = choice.get("logprobs")
    content = lp.get("content") if isinstance(lp, dict) else None
    usage = data.get("usage") or {}
    return Completion(
        text=text,
        n_tokens=int(usage.get("completion_tokens", 0)),
        finish_reason=choice.get("finish_reason"),
        seed=seed,
        logprobs=content,
    )
