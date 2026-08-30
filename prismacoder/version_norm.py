"""Prisma version normalization (report: corpus spans v4/v5/v6/v7 syntax).

``normalize_to_v7`` rewrites legacy schema syntax to the current dialect so the
verification sandbox sees a homogeneous target:

* ``provider = "prisma-client-js"``   -> ``provider = "prisma-client"`` + output path
  (Prisma 7's Rust-free client generates to a project path).
* ``url = env("DATABASE_URL")``       kept.
* stale ``previewFeatures``           dropped (e.g. "views" GA'd, "relationJoins" GA'd).
* ``@db.` types for sqlite            dropped (sqlite ignores native types and
                                      older corpora abuse them).
* ``engineType = "library"``          dropped (library-only in v7).

Normalizers are pure functions over text, recorded in the record's
``meta.tags`` as ``v7-normalized`` so provenance survives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PREVIEW_FEATURES_STALE = {
    "views",
    "relationJoins",
    "fullTextSearchPostgres",
    "fullTextSearchMySQL",
    "metrics",
    "dataProxy",
    "extendedIndexes",
    "referentialActions",
    "fieldReference",
    "objectRelationalMapping",
    "jdbcSqlServer",
}


@dataclass
class NormalizeReport:
    changes: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def _rewrite_generator(schema: str, report: NormalizeReport) -> str:
    def repl(m: re.Match[str]) -> str:
        body = m.group(1)
        if 'provider = "prisma-client"' in body or "provider = 'prisma-client'" in body:
            return m.group(0)
        report.changes.append("generator: prisma-client-js -> prisma-client (+output)")
        body = re.sub(r'provider\s*=\s*"prisma-client-js"', 'provider = "prisma-client"', body)
        if not re.search(r"output\s*=", body):
            body = body.rstrip().rstrip("}")
            body += '\n  output  = "../app/generated/prisma"\n}'
        # drop binaryTargets/engineType lines (v7 library-only, client no longer uses them)
        body = re.sub(r"\n\s*(binaryTargets|engineType)\s*=.*", "", body)
        return f"generator client {{{body}}}" if not body.startswith("\n") else f"generator client {body}"

    return re.sub(r"generator\s+\w*\s*\{(.*?)\}", repl, schema, flags=re.S)


def _drop_stale_preview(schema: str, report: NormalizeReport) -> str:
    def repl(m: re.Match[str]) -> str:
        feats = re.findall(r'"([^"]+)"', m.group(0))
        kept = [f for f in feats if f not in PREVIEW_FEATURES_STALE]
        dropped = [f for f in feats if f in PREVIEW_FEATURES_STALE]
        if dropped:
            report.changes.append(f"previewFeatures: dropped {dropped}")
        if not kept:
            return ""
        inner = "".join(f'\n  "{f}"' for f in kept)
        return f"previewFeatures = [{inner}\n]"

    return re.sub(r"previewFeatures\s*=\s*\[.*?\]", repl, schema, flags=re.S)


def _strip_sqlite_native_types(schema: str, report: NormalizeReport) -> str:
    if "@db." not in schema:
        return schema
    report.changes.append("stripped @db.* native type attributes")
    return re.sub(r"\s+@db\.\w+", "", schema)


def normalize_to_v7(schema: str) -> tuple[str, NormalizeReport]:
    report = NormalizeReport(changes=[])
    out = schema
    out = _drop_stale_preview(out, report)
    out = _rewrite_generator(out, report)
    out = _strip_sqlite_native_types(out, report)
    out = _drop_datasource_url(out, report)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out, report


def _drop_datasource_url(schema: str, report: NormalizeReport) -> str:
    """Prisma 7 removed datasource.url: the connection URL lives in
    prisma.config.ts. Strip the line and record the change."""
    if re.search(r'(?m)^\s*url\s*=\s*env\("DATABASE_URL"\)\s*$', schema):
        report.changes.append("datasource: url moved to prisma.config.ts (v7 dialect)")
    return re.sub(r'(?m)^\s*url\s*=\s*env\("DATABASE_URL"\)\s*\n', "", schema)


def detect_version(schema: str) -> str:
    """Best-effort corpus version tag for provenance (pre-normalization)."""
    if 'provider = "prisma-client"' in schema or 'provider = "prisma-client"' in schema:
        return "v7-style"
    if "previewFeatures" in schema:
        feats = set(re.findall(r'"([^"]+)"', schema))
        if feats & {"views", "relationJoins"}:
            return "v6"
        if feats & {"fullTextSearchPostgres", "extendedIndexes"}:
            return "v5"
        return "v4+"
    if re.search(r"provider\s*=\s*[\"']prisma-client-js", schema):
        return "v4/v5"
    return "unknown"
