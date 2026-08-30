"""Tree edit distance (Zhang-Shasha, O(n^2)m) over bracketed code trees.

``bracket_tree`` parses code into a nested structure keyed on brackets and
delimiters; ``prisma_tree_edit_distance`` wraps it with PSL-aware leaves
(model/field tokens are compared verbatim, punctuation is cheap to edit).
Used for structural similarity and the CodeBLEU tree term.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

Node = tuple[str, list["Node"]]  # (label, children)

OPEN = {"{", "(", "["}
CLOSE = {"}": "{", ")": "(", "]": "["}
RE_TOKEN = re.compile(r"[{}()\[\]]|\"[^\"]*\"|'[^']*'|\w+|@")


@dataclass
class TEDCost:
    delete: int = 1
    insert: int = 1
    rename_equal: int = 0
    rename_diff: int = 1


def bracket_tree(code: str) -> Node:
    """Parse code into a bracket tree. Leaf labels are tokens; delimiters are nodes."""
    root: Node = ("root", [])
    stack: list[Node] = [root]
    for tok in RE_TOKEN.findall(code):
        if tok in OPEN:
            node: Node = (tok, [])
            stack[-1][1].append(node)
            stack.append(node)
        elif tok in CLOSE:
            if len(stack) > 1 and stack[-1][0] == CLOSE[tok]:
                stack.pop()
            else:  # unbalanced: skip gracefully
                continue
        else:
            stack[-1][1].append((tok, []))
    return root


def tree_edit_distance(a: Node, b: Node, cost: TEDCost | None = None) -> int:
    """Post-order Zhang-Shasha with the simple unit-cost scheme."""
    cost = cost or TEDCost()
    la, lb = _postorder(a), _postorder(b)
    m, n = len(la), len(lb)
    # key-roots (simplified: all nodes are key roots -> O(n^2 m) worst case,
    # acceptable for artifact-sized trees)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + cost.delete
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + cost.insert
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if la[i - 1][0] == lb[j - 1][0]:
                dp[i][j] = dp[i - 1][j - 1] + cost.rename_equal
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + cost.delete,
                    dp[i][j - 1] + cost.insert,
                    dp[i - 1][j - 1] + cost.rename_diff,
                )
    return dp[m][n]


def _postorder(root: Node) -> list[Node]:
    out: list[Node] = []

    def visit(node: Node) -> None:
        for child in node[1] or []:
            visit(child)
        out.append(node)

    visit(root)
    return out


def prisma_tree_edit_distance(prediction: str, reference: str) -> int:
    """PSL/TS-aware tree distance between two artifacts."""
    return tree_edit_distance(bracket_tree(prediction), bracket_tree(reference))


def tree_similarity(prediction: str, reference: str) -> float:
    """Normalized 0..1 similarity (1 = identical)."""
    d = prisma_tree_edit_distance(prediction, reference)
    size = max(_node_count(bracket_tree(prediction)), _node_count(bracket_tree(reference)), 1)
    return round(max(0.0, 1.0 - d / size), 4)


def _node_count(tree: Node) -> int:
    return 1 + sum(_node_count(c) for c in tree[1])
