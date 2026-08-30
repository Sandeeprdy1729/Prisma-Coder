"""Custom collators + length-grouped helpers for the PrismaCoder trainers.

torch is imported lazily so config/data utilities work on CPU-only CI.
"""

from __future__ import annotations

from typing import Any


def _pad_sequence(sequences, batch_first: bool, padding_value: int):
    from torch.nn.utils.rnn import pad_sequence as _pad

    return _pad(sequences, batch_first=batch_first, padding_value=padding_value)


class CompletionOnlyCollator:
    """Pads pre-tokenized {input_ids, labels} examples; labels are -100 on
    prompt spans (set by data.to_chat_example)."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict:
        import torch

        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        input_ids = _pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = _pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class PackingCollator:
    """Pads pre-packed sequences of max_seq_len (wrapped packing)."""

    def __init__(self, pad_token_id: int, max_seq_len: int = 4096):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, features: list[dict[str, Any]]) -> dict:
        import torch

        input_ids = [torch.tensor(f["input_ids"][: self.max_seq_len], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"][: self.max_seq_len], dtype=torch.long) for f in features]
        input_ids = _pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = _pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def pack_examples(
    examples: list[dict[str, Any]], max_seq_len: int, pad_token_id: int
) -> list[dict[str, Any]]:
    """Greedy first-fit packing of tokenized examples into max_seq_len rows (pure python)."""
    packed: list[dict[str, Any]] = []
    cur_ids: list[int] = []
    cur_labels: list[int] = []
    bos = pad_token_id  # separator token between packed documents
    for ex in examples:
        ids, labels = list(ex["input_ids"]), list(ex["labels"])
        if len(ids) > max_seq_len:
            ids, labels = ids[:max_seq_len], labels[:max_seq_len]
        if len(cur_ids) + len(ids) + 1 > max_seq_len:
            if cur_ids:
                packed.append({"input_ids": cur_ids, "labels": cur_labels})
            cur_ids, cur_labels = [], []
        cur_ids += ids + [bos]
        cur_labels += labels + [-100]
    if cur_ids:
        packed.append({"input_ids": cur_ids, "labels": cur_labels})
    return packed
