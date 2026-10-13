"""Client parsing tests against a mocked llama-server. No process, no network."""

import json
from collections.abc import Callable

import httpx

from modeldetective.serve import LlamaClient, LlamaServer, SamplerConfig


def _client_with(handler: Callable[[httpx.Request], httpx.Response]) -> LlamaClient:
    c = LlamaClient("http://test")
    c._http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return c


def test_sampler_request_fields() -> None:
    req = SamplerConfig(temperature=1.0, top_k=0, max_tokens=8).as_request()
    assert req["temperature"] == 1.0
    assert req["top_k"] == 0
    assert req["max_tokens"] == 8
    assert req["repeat_penalty"] == 1.0


def test_complete_parses_text_tokens_and_logprobs() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "text": "0101",
                        "finish_reason": "length",
                        "logprobs": {
                            "content": [
                                {"token": "01", "logprob": -0.5, "top_logprobs": []},
                                {"token": "01", "logprob": -0.7, "top_logprobs": []},
                            ]
                        },
                    }
                ],
                "usage": {"completion_tokens": 2, "prompt_tokens": 5},
            },
        )

    with _client_with(handler) as c:
        out = c.complete("p", SamplerConfig(max_tokens=4), seed=7, n_logprobs=3)
    assert seen["path"] == "/v1/completions"
    assert seen["body"]["seed"] == 7
    assert seen["body"]["logprobs"] == 3
    assert out.text == "0101"
    assert out.n_tokens == 2
    assert out.seed == 7
    assert out.logprobs is not None and len(out.logprobs) == 2


def test_chat_parses_message_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["logprobs"] is True and body["top_logprobs"] == 5
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1},
            },
        )

    with _client_with(handler) as c:
        out = c.chat([{"role": "user", "content": "hi"}], SamplerConfig(), n_logprobs=5)
    assert out.text == "42"
    assert out.n_tokens == 1
    assert out.finish_reason == "stop"
    assert out.logprobs is None


def test_server_command_cpu_policy(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s = LlamaServer(model_path=tmp_path / "m.gguf", port=9000, n_gpu_layers=0, device="none")
    cmd = s.command()
    assert cmd[cmd.index("--device") + 1] == "none"
    assert cmd[cmd.index("-ngl") + 1] == "0"
    assert s.base_url == "http://127.0.0.1:9000"
