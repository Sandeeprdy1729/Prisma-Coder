"""Dataset formatting + tokenization for SFT/DPO/CPT (training/trainer/data.py).

Converts PrismaForge records into chat-format training examples:
  system  -> constants.SYSTEM_PROMPT
  user    -> instruction + schema context (+ input / repo files)
  assistant -> reference artifact (loss applied here only, unless packing)

Also implements the epoch-2/3 soft-ordering mixture (report 12.3) and the
DPO pair formatting for TRL's DPOTrainer.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Iterator

from prismacoder.constants import SYSTEM_PROMPT
from prismacoder.io_utils import read_jsonl

log = logging.getLogger(__name__)

CODE_FENCE = "```"


def build_user_message(rec: dict[str, Any]) -> str:
    """Instruction + grounded context (schema, optional files/input)."""
    task = rec["task"]
    ctx = task.get("context", {})
    parts: list[str] = [task["instruction"]]
    schema = (ctx.get("schema") or "").strip()
    if schema:
        parts.append(f"Here is the current schema.prisma:\n\n{CODE_FENCE}prisma\n{schema}\n{CODE_FENCE}")
    files = ctx.get("files") or []
    for f in files[:6]:
        name = f.get("path", "file.ts")
        parts.append(f"File {name}:\n\n{CODE_FENCE}ts\n{f.get('content', '')}\n{CODE_FENCE}")
    if task.get("input"):
        parts.append(f"Task input:\n\n{CODE_FENCE}ts\n{task['input']}\n{CODE_FENCE}")
    return "\n\n".join(parts)


def build_assistant_message(rec: dict[str, Any]) -> str:
    ref = (rec["task"].get("reference") or "").strip()
    # normalize to a single fenced artifact for schema/TS outputs
    if ref and not ref.startswith(CODE_FENCE) and not ref.startswith("//"):
        kind = "prisma" if "model " in ref and "datasource" in ref else "ts"
        ref = f"{CODE_FENCE}{kind}\n{ref}\n{CODE_FENCE}"
    return ref


def to_chat_example(
    rec: dict[str, Any], tokenizer: Any, system_prompt: str = SYSTEM_PROMPT, with_reasoning: bool = False
) -> dict[str, Any]:
    """Render one record into tokenized prompt/completion (TRL SFT format)."""
    user = build_user_message(rec)
    reasoning = rec.get("reasoning", {})
    assistant = build_assistant_message(rec)
    if with_reasoning and reasoning.get("plan_span"):
        assistant = f"Plan:\n{reasoning['plan_span']}\n\n{assistant}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    completion_ids = full_ids[len(prompt_ids) :] if full_ids[: len(prompt_ids)] == prompt_ids else full_ids
    return {
        "input_ids": full_ids,
        "labels": [-100] * len(prompt_ids) + list(completion_ids),
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
    }


def to_dpo_example(rec: dict[str, Any], tokenizer: Any, system_prompt: str = SYSTEM_PROMPT) -> dict[str, Any]:
    """Format a PrismaForge DPO pair for TRL DPOTrainer."""
    schema = rec.get("context_schema", "")
    user = rec["prompt"]
    if schema:
        user += f"\n\n{CODE_FENCE}prisma\n{schema}\n{CODE_FENCE}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "chosen": rec["chosen"], "rejected": rec["rejected"]}


@dataclass
class SoftOrderingMixer:
    """Epoch-dependent mixture (report 12.3): epoch 1 oversamples L1 (60%),
    epoch 2 balances, epoch 3 oversamples L3 + repair tasks. Persona tags are
    varied per epoch to avoid persona overfitting."""

    seed: int = 20260830

    def weights_for_epoch(self, epoch: int) -> dict[str, float]:
        if epoch <= 1:
            return {"L1": 0.60, "L2": 0.30, "L3": 0.10}
        if epoch == 2:
            return {"L1": 0.34, "L2": 0.40, "L3": 0.26}
        return {"L1": 0.20, "L2": 0.35, "L3": 0.45}

    def stream(self, records: list[dict[str, Any]], epochs: int) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.seed)
        by_level: dict[str, list[dict]] = {}
        for rec in records:
            by_level.setdefault(rec["task"].get("difficulty", "L1"), []).append(rec)
        for epoch in range(1, epochs + 1):
            weights = self.weights_for_epoch(epoch)
            levels = [lv for lv in ("L1", "L2", "L3") if by_level.get(lv)]
            if not levels:
                yield from records
                continue
            pool_weights = [weights.get(lv, 0.1) for lv in levels]
            epoch_order: list[dict] = []
            target = sum(len(by_level[lv]) for lv in levels)
            level_rng = random.Random(self.seed + epoch)
            for _ in range(target):
                lv = level_rng.choices(levels, weights=pool_weights)[0]
                epoch_order.append(by_level[lv][level_rng.randrange(len(by_level[lv]))])
            rng.shuffle(epoch_order)
            for rec in epoch_order:
                yield rec


def load_sft_records(
    train_file: str, validation_file: str | None = None, limit: int | None = None
) -> tuple[list[dict], list[dict]]:
    train = read_jsonl(train_file)
    if limit:
        train = train[:limit]
    val = read_jsonl(validation_file) if validation_file else []
    log.info("loaded %d train / %d val records", len(train), len(val))
    return train, val


def iter_cpt_corpus(path: str, mixture: dict[str, float] | None = None) -> Iterator[dict[str, str]]:
    """Stream {text} rows for CPT with optional family reweighting."""
    rows = read_jsonl(path)
    rng = random.Random(20260830)
    if mixture:
        by_family: dict[str, list[str]] = {}
        for r in rows:
            fam = r.get("family", "code_other")
            by_family.setdefault(fam, []).append(r.get("text", ""))
        families = list(mixture)
        weights = [mixture[f] for f in families]
        total = sum(len(v) for v in by_family.values())
        for _ in range(total):
            fam = rng.choices(families, weights=weights)[0]
            pool = by_family.get(fam) or []
            if pool:
                yield {"text": pool[rng.randrange(len(pool))]}
    else:
        rng.shuffle(rows)
        for r in rows:
            yield {"text": r.get("text", "")}


def dump_jsonl_preview(records: list[dict], path: str, n: int = 5) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records[:n]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
