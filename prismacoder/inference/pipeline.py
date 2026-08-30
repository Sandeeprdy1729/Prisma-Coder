"""PrismaCoderPipeline: load model -> accept NL -> generate -> validate ->
repair -> return final code.

Works with:
- a merged HF model dir (``--model runs/.../merged``)
- base + adapter (``--adapter``)
- the published hub checkpoint (``prismacoder/PrismaCoder-3B``)

The heavy transformers stack is imported lazily; the pipeline exposes
``from_local()`` / ``from_hub()`` factories and a ``generate`` callable that
plugs into :class:`prismacoder.inference.repair.RepairLoop`. When a Node
toolchain is available, pass ``use_full_gates=True`` to repair against the
real V1-V5 chain instead of the static validator.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from prismacoder.constants import SYSTEM_PROMPT
from prismacoder.inference.repair import RepairLoop, RepairResult
from prismacoder.verification.sandbox import node_available
from prismacoder.verification.verifier import LightValidator, Verifier

RE_FENCE = re.compile(r"```(?:prisma|ts|typescript)?\n(.*?)```", re.S)


@dataclass
class GenerationStats:
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 0


class PrismaCoderPipeline:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        system_prompt: str = SYSTEM_PROMPT,
        use_full_gates: bool = False,
        repair_attempts: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.repair_attempts = repair_attempts
        self.light = LightValidator()
        self.full_verifier: Verifier | None = None
        if use_full_gates:
            node, _ = node_available()
            if node:
                self.full_verifier = Verifier(run_install=False)
        self.repair = RepairLoop(self.light, max_attempts=repair_attempts, full_verifier=self.full_verifier)
        self.last_stats = GenerationStats()

    # -- factories -----------------------------------------------------------------
    @classmethod
    def from_local(cls, model_dir: str, **kw: Any) -> PrismaCoderPipeline:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return cls(model, tokenizer, **kw)

    @classmethod
    def from_base_and_adapter(cls, base: str, adapter: str, **kw: Any) -> PrismaCoderPipeline:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        return cls(model, tokenizer, **kw)

    # -- core ------------------------------------------------------------------------
    def _build_messages(self, instruction: str, schema: str) -> list[dict[str, str]]:
        content = instruction
        if schema.strip():
            content += f"\n\n```prisma\n{schema.strip()}\n```"
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}]

    def raw_generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        started = time.monotonic()
        import torch

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-4),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        self.last_stats.latency_ms = int((time.monotonic() - started) * 1000)
        self.last_stats.prompt_tokens = int(inputs["input_ids"].shape[1])
        self.last_stats.completion_tokens = int(out.shape[1] - inputs["input_ids"].shape[1])
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def extract_code(self, text: str) -> str:
        """Pull the code out of a fenced response (fall back to raw text)."""
        blocks = RE_FENCE.findall(text)
        return blocks[0].strip() if blocks else text.strip()

    def run(self, instruction: str, schema: str = "", *, repair: bool = True) -> RepairResult:
        """NL (+optional schema) -> final, validated artifact."""
        messages = self._build_messages(instruction, schema)
        first = self.extract_code(self.raw_generate(messages))
        if not repair:
            verdict = self.light.validate(schema, first)
            return RepairResult(
                artifact=first, status="failed" if verdict.status == "failed" else "unverified"
            )
        return self.repair.run(instruction, schema, self.raw_generate, first_artifact=first)

    # -- streaming (token iterator for the SSE API) ------------------------------------
    def stream(self, instruction: str, schema: str = ""):
        from threading import Thread

        from transformers import TextIteratorStreamer

        messages = self._build_messages(instruction, schema)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-4),
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        import torch

        thread = Thread(target=lambda: torch.inference_mode()(self.model.generate)(**kwargs))
        thread.start()
        return streamer
