"""JSONL / JSON IO helpers used by every pipeline stage. Stdlib only."""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a .jsonl (or .jsonl.gz) file into a list of dicts. Skips blank lines."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    out: list[dict[str, Any]] = []
    with opener(p, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:  # pragma: no cover - corrupt shard guard
                raise ValueError(f"{p}:{i}: invalid JSON ({e})") from e
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL file without loading everything into memory."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]], *, atomic: bool = True) -> int:
    """Write records as JSONL. Atomic by default (tmp file + os.replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    buf = io.StringIO()
    for rec in records:
        buf.write(json.dumps(rec, ensure_ascii=False, sort_keys=False) + "\n")
        n += 1
    if atomic:
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
        os.replace(tmp, p)
    else:
        p.write_text(buf.getvalue(), encoding="utf-8")
    return n


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | Path, obj: Any, *, atomic: bool = True, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=indent) + "\n"
    if atomic:
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, p)
    else:
        p.write_text(text, encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_lines(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rb") as f:
        return sum(1 for _ in f)


def shard_path(base_dir: str | Path, name: str, shard: int, shards: int) -> Path:
    """Deterministic shard filename: pool/shard-00003-of-00100.jsonl"""
    return Path(base_dir) / f"{name}" / f"shard-{shard:05d}-of-{shards:05d}.jsonl"
