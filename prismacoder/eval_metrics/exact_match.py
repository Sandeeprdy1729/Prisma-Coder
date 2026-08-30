"""Exact match on normalized artifacts (report 11.1)."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_COMMENTS_TS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_COMMENTS_PSL = re.compile(r"(?m)^//.*$|^///.*$")
_TRAILING_SEMI = re.compile(r";\s*$")


def normalized_artifact(text: str, *, strip_comments: bool = True) -> str:
    """Whitespace/comment-insensitive normalization for exact-match scoring."""
    t = text.strip()
    if strip_comments:
        t = _COMMENTS_TS.sub("", t)
        t = _COMMENTS_PSL.sub("", t)
    t = _WS.sub(" ", t)
    return t.strip()


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalized_artifact(prediction) == normalized_artifact(reference) else 0.0
