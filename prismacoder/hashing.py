"""Hashing + provenance utilities (report 9.3: provenance_hash on every record)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    """Stable hash of a JSON-serializable object (sorted keys, no whitespace)."""
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def normalize_text(text: str) -> str:
    """Whitespace-insensitive normalization used for exact dedup."""
    return re.sub(r"\s+", " ", text).strip()


def exact_hash(text: str) -> str:
    """Exact-duplicate key: sha256 over whitespace-normalized text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    """Word-level k-shingles."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def model_graph_signature(schema_text: str) -> str:
    """Isomorphism-tolerant signature of a PSL schema.

    Extracts the model block graph (model names, relation edges, cardinalities)
    and renames models/fields to canonical symbols (M1..Mn) so that two
    schemas differing only by identifiers collide. Used for the third dedup
    granularity (AST-cluster signatures, report 9.2).
    """
    from prismacoder.psl import parse_models  # local import to avoid cycle

    models = parse_models(schema_text)
    if not models:
        return ""
    name_map: dict[str, str] = {}
    edges: list[str] = []
    field_profiles: list[str] = []
    for idx, m in enumerate(sorted(models, key=lambda x: x.name)):
        name_map[m.name] = f"M{idx + 1}"
    for m in sorted(models, key=lambda x: x.name):
        rels = []
        for f in m.relation_fields():
            if f.type in name_map:
                rels.append(f"{f.name}:{name_map[f.type]}" + ("[]" if f.is_list else ""))
        rels.sort()
        edges.append(f"{name_map[m.name]}({','.join(rels)})")
        # profile = multiset of scalar types with optionality/list flags
        prof = ",".join(
            sorted(
                f"{f.type}{'?' if f.is_optional else ''}{'L' if f.is_list else ''}" for f in m.scalar_fields()
            )
        )
        field_profiles.append(prof)
    import hashlib

    return hashlib.sha1(("|".join(edges) + "#" + "|".join(sorted(field_profiles))).encode()).hexdigest()
