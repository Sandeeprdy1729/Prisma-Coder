"""BLEU (corpus + sentence) - dependency-free, standard smoothing-free form."""

from __future__ import annotations

import re
from collections import Counter
from math import exp, log

_TOKEN = re.compile(r"\w+|[\{\}\(\)\[\];,.<>=+\-*/!?]|\"[^\"]*\"|'[^']*'")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _clipped_precision(cand_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    c = Counter(_ngrams(cand_tokens, n))
    r = Counter(_ngrams(ref_tokens, n))
    if not c:
        return 0.0
    clipped = sum(min(count, r.get(g, 0)) for g, count in c.items())
    total = sum(c.values())
    return clipped / total if total else 0.0


def sentence_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    c, r = tokenize(prediction), tokenize(reference)
    if not c or not r:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        p = _clipped_precision(c, r, n)
        if p == 0.0:
            # brevity-safe: bail out with a small floor rather than smoothing games
            return 0.0
        precisions.append(p)
    bp = 1.0 if len(c) > len(r) else exp(1 - len(r) / max(len(c), 1))
    log_avg = sum(log(p) for p in precisions) / max_n
    return bp * exp(log_avg)


def corpus_bleu(predictions: list[str], references: list[str], max_n: int = 4) -> float:
    """Multi-bleu style corpus score: sum clipped counts over the corpus."""
    if not predictions:
        return 0.0
    log_ps = []
    for n in range(1, max_n + 1):
        num = den = 0
        for p, r in zip(predictions, references, strict=False):
            ct, rt = tokenize(p), tokenize(r)
            c = Counter(_ngrams(ct, n))
            rr = Counter(_ngrams(rt, n))
            num += sum(min(count, rr.get(g, 0)) for g, count in c.items())
            den += sum(c.values())
        if den == 0 or num == 0:
            return 0.0
        log_ps.append(log(num / den))
    total_c = sum(len(tokenize(p)) for p in predictions)
    total_r = sum(len(tokenize(r)) for r in references)
    bp = 1.0 if total_c > total_r else exp(1 - total_r / max(total_c, 1))
    return bp * exp(sum(log_ps) / max_n)
