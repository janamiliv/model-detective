"""Sample from an onnxruntime-genai model folder with the project's sampler settings.

Used for the H3 method-fingerprint track (ONNX INT8/INT4 vs GGUF K-quants). Note the runtime
differs from llama-server, so method and runtime are confounded here; DECISIONS.md says so.

Sampling: genai defaults are do_sample=False, top_k=50, top_p=0.0. For plain temperature-1
multinomial sampling we must set do_sample=True, top_k=0, top_p=1.0 (top_k=0 with top_p=0.0
collapses to argmax). The chat template is rendered with the HF tokenizer because genai's
apply_chat_template takes no kwargs and we need enable_thinking=False.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnxruntime_genai as og
from transformers import AutoTokenizer


@dataclass
class OnnxSampler:
    model_dir: Path
    hf_tokenizer_dir: Path
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 88

    def __post_init__(self) -> None:
        self.model = og.Model(str(self.model_dir))
        self.tok = og.Tokenizer(self.model)
        self.hf_tok = AutoTokenizer.from_pretrained(str(self.hf_tokenizer_dir))

    def render(self, user: str, system: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        return self.hf_tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def sample(self, prompt_text: str, seed: int) -> tuple[str, int]:
        ids = self.tok.encode(prompt_text)
        params = og.GeneratorParams(self.model)
        params.set_search_options(
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            random_seed=seed,
            max_length=len(ids) + self.max_new_tokens,
        )
        gen = og.Generator(self.model, params)
        gen.append_tokens(ids)
        out: list[int] = []
        while not gen.is_done():
            gen.generate_next_token()
            out.append(int(gen.get_next_tokens()[0]))
        return self.tok.decode(out), len(out)
