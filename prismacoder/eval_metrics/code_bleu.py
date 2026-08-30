"""CodeBLEU: n-gram BLEU + syntax (keyword/AST token) match + dataflow blend.

Implements the practical variant used for code tasks: BLEU over code tokens,
a weighted syntax match over language keywords + PSL/Prisma API tokens, and a
 lightweight structural term via tree edit distance on bracketed structure.
(Reference: CodeXGLUE CodeBLEU, adapted to PSL/TS without tree-sitter.)
"""

from __future__ import annotations

import re

from prismacoder.eval_metrics.bleu import _clipped_precision, tokenize  # noqa: F401
from prismacoder.eval_metrics.tree_edit_distance import bracket_tree, tree_edit_distance

TS_KEYWORDS = {
    "import",
    "export",
    "from",
    "const",
    "let",
    "var",
    "function",
    "return",
    "await",
    "async",
    "if",
    "else",
    "for",
    "while",
    "class",
    "new",
    "try",
    "catch",
    "throw",
    "interface",
    "type",
    "extends",
    "implements",
    "public",
    "private",
    "readonly",
    "static",
    "default",
    "of",
    "in",
}
PRISMA_API = {
    "prisma",
    "findMany",
    "findUnique",
    "findFirst",
    "create",
    "createMany",
    "update",
    "updateMany",
    "delete",
    "deleteMany",
    "upsert",
    "aggregate",
    "groupBy",
    "count",
    "$transaction",
    "$queryRaw",
    "$executeRaw",
    "include",
    "select",
    "where",
    "data",
    "orderBy",
    "take",
    "skip",
    "cursor",
    "distinct",
    "connect",
    "connectOrCreate",
    "disconnect",
    "increment",
    "decrement",
    "model",
    "enum",
    "datasource",
    "generator",
    "@id",
    "@unique",
    "@default",
    "@relation",
    "@map",
    "@@id",
    "@@unique",
    "@@index",
    "@@map",
    "String",
    "Int",
    "Float",
    "Boolean",
    "DateTime",
    "Json",
    "BigInt",
    "Decimal",
}


def _syntax_tokens(text: str) -> set[str]:
    toks = set(re.findall(r"\w+|\$\w+|@{1,2}\w+", text))
    return toks & (TS_KEYWORDS | PRISMA_API)


def code_bleu(
    prediction: str, reference: str, *, alpha: float = 0.5, beta: float = 0.2, gamma: float = 0.3
) -> float:
    """alpha: n-gram weight, beta: syntax weight, gamma: tree weight (sum to 1)."""
    if not prediction.strip() or not reference.strip():
        return 0.0
    # 1. n-gram BLEU (4-gram, no smoothing)
    from prismacoder.eval_metrics.bleu import sentence_bleu

    ngram = sentence_bleu(prediction, reference)
    # 2. syntax match over keywords + prisma API surface
    p_syn, r_syn = _syntax_tokens(prediction), _syntax_tokens(reference)
    if p_syn or r_syn:
        syn = len(p_syn & r_syn) / max(len(p_syn | r_syn), 1)
    else:
        syn = 1.0
    # 3. tree match on bracketed structure
    try:
        p_tree = bracket_tree(prediction)
        r_tree = bracket_tree(reference)
        dist = tree_edit_distance(p_tree, r_tree)
        size = max(_tree_size(p_tree), _tree_size(r_tree), 1)
        tree_match = max(0.0, 1.0 - dist / size)
    except RecursionError:
        tree_match = 0.0
    return round(alpha * ngram + beta * syn + gamma * tree_match, 4)


def _tree_size(tree: tuple) -> int:
    count = 1
    for child in tree[1] or []:
        count += _tree_size(child)
    return count
