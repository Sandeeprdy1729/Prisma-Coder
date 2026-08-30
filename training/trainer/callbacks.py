"""Trainer callbacks: gate-aligned sanity generation + gate-profile logging.

The sanity callback generates a small Prisma artifact every N steps and runs
the static symbol-grounding check (LightValidator) so the training log shows a
proxy for the gate profile that actually decides quality (report 11: 'per-gate
reporting so a score decline is interpretable').
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from prismacoder.constants import SYSTEM_PROMPT
from prismacoder.verification.verifier import LightValidator

log = logging.getLogger(__name__)


@dataclass
class SanitySpec:
    enabled: bool = True
    every_steps: int = 500
    num_examples: int = 2
    max_new_tokens: int = 512
    temperature: float = 0.2


def make_sanity_callback(tokenizer, model, sanity_examples: list[dict], spec: SanitySpec):
    """Returns a TrainerCallback; heavy imports are deferred to training time."""
    try:
        from transformers import TrainerCallback
    except ImportError as e:  # pragma: no cover
        raise ImportError("transformers is required: pip install -e '.[training]'") from e

    light = LightValidator()

    class PrismaSanityCallback(TrainerCallback):
        def __init__(self):
            self.last_fire = -spec.every_steps

        def on_step_end(self, args, state, control, **kwargs):
            if not spec.enabled or state.global_step - self.last_fire < spec.every_steps:
                return
            self.last_fire = state.global_step
            if not sanity_examples:
                return
            model.eval()
            static_pass = 0
            try:
                for rec in sanity_examples[: spec.num_examples]:
                    schema = rec["task"]["context"]["schema"]
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": rec["task"]["instruction"] + "\n\n```prisma\n" + schema + "\n```",
                        },
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    out = model.generate(
                        **inputs,
                        max_new_tokens=spec.max_new_tokens,
                        do_sample=True,
                        temperature=spec.temperature,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                    verdict = light.validate(schema, text)
                    static_pass += 1 if verdict.status != "failed" else 0
                    log.info(
                        "sanity step=%s artifact_chars=%s static_pass=%s",
                        state.global_step,
                        len(text),
                        verdict.status,
                    )
            except Exception:  # noqa: BLE001 - never break a training run
                log.exception("sanity generation failed")
            finally:
                model.train()
            if sanity_examples:
                log.info(
                    "sanity static pass rate: %d/%d",
                    static_pass,
                    min(spec.num_examples, len(sanity_examples)),
                )

    return PrismaSanityCallback()


def resolve_dtype(cfg: dict):
    import torch

    return torch.bfloat16 if cfg.get("torch_dtype", "bfloat16") == "bfloat16" else torch.float16
