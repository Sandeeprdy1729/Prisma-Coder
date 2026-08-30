"""Prisma artifact extractor (Phase 1).

Turns raw crawler snapshots into structured extraction records:

    raw snapshot (files/) ──> ExtractionRecord per schema
        - schema text + PSL parse
        - client call sites grouped by operation family:
            crud (findMany/create/update/delete/upsert/createMany/updateMany/deleteMany)
            relations (include/select/connect/connectOrCreate/nested payloads)
            transactions ($transaction interactive + sequential)
            aggregations (aggregate/count/findFirst...)
            group_by (groupBy + having/_count/_sum/_avg/_min/_max)
            raw_sql ($queryRaw/$executeRaw + tagged templates)
        - surrounding context layer: service / repository / API / resolver
          (detected from the file path + class/function names)

The extractor is deterministic and dependency-free (regex + brace matching);
tree-sitter is an optional refinement, not a requirement. Extraction never
mutates the raw snapshot.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

from prismacoder.hashing import sha256_text
from prismacoder.psl import parse_schema

log = logging.getLogger("prismaforge.extractor")

# --- operation families (client-call census, report 4.3) --------------------
CRUD_OPS = (
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
)
AGG_OPS = ("aggregate", "count", "groupBy")
RAW_OPS = ("$queryRaw", "$executeRaw", "$queryRawUnsafe", "$executeRawUnsafe")

RE_CALL = re.compile(r"\b(prisma|db|tx|p)\s*\.\s*(\w+)\s*\.\s*(\w+)\s*(?:\(|<)")
RE_TRANSACTION = re.compile(r"\$transaction\s*\(")
RE_INCLUDE = re.compile(r"\binclude\s*:\s*\{")
RE_SELECT = re.compile(r"\bselect\s*:\s*\{")
RE_CONNECT = re.compile(r"\b(connect|connectOrCreate)\s*:\s*[\{\[\"']")
RE_NESTED_WRITE = re.compile(r"\b(create|update|upsert|delete|connect|disconnect|set)\s*:\s*\{")
RE_RAW = re.compile(r"\$(queryRaw|executeRaw)(Unsafe)?\s*(`|\()")
RE_GROUPBY = re.compile(r"\.groupBy\s*(?:<[^>]*>)?\s*\(")
RE_ORDERBY = re.compile(r"\borderBy\s*:")
RE_CURSOR = re.compile(r"\bcursor\s*:\s*\{")
RE_DISTINCT = re.compile(r"\bdistinct\s*:")
RE_WHERE = re.compile(r"\bwhere\s*:\s*\{")
RE_CLASS_REPO = re.compile(r"\bclass\s+(\w*(Repository|Repo|DAO|Store)\w*)\b")
RE_SERVICE = re.compile(r"\bclass\s+(\w*(Service|Manager|UseCase)\w*)\b")
RE_ROUTER = re.compile(
    r"\b(?:publicProcedure|protectedProcedure|router\s*\(|@(?:Get|Post|Put|Patch|Delete)\()"
)
RE_GRAPHQL = re.compile(r"\b(?:Query|Mutation)\s*:\s*\{|@(?:Resolver|Args|Field)\(")

MIGRATION_SECTIONS = re.compile(r"(?im)^(--\s*(Create|Alter|Drop)[^\n]*|--\s+AddForeignKey|--\s+CreateIndex)")


@dataclass
class CallSite:
    op: str  # e.g. findMany
    model_hint: str  # lowercased client property (e.g. "user")
    family: str  # crud | aggregation | group_by | raw_sql | transaction
    file: str
    line: int
    snippet: str


@dataclass
class ExtractionRecord:
    repo: str
    track: str
    stack: str
    license: str
    snapshot_ref: str
    schema_path: str
    schema: str
    schema_hash: str
    models: list[str]
    migrations: list[str] = dc_field(default_factory=list)
    call_sites: list[CallSite] = dc_field(default_factory=list)
    layers: dict[str, list[str]] = dc_field(
        default_factory=lambda: {"service": [], "repository": [], "api": [], "graphql": []}
    )
    features: dict[str, bool] = dc_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "track": self.track,
            "stack": self.stack,
            "license": self.license,
            "snapshot_ref": self.snapshot_ref,
            "schema_path": self.schema_path,
            "schema": self.schema,
            "schema_hash": self.schema_hash,
            "models": self.models,
            "migrations": self.migrations,
            "call_sites": [cs.__dict__ for cs in self.call_sites],
            "layers": self.layers,
            "features": self.features,
        }


def _families(code: str) -> dict[str, bool]:
    return {
        "include": bool(RE_INCLUDE.search(code)),
        "select": bool(RE_SELECT.search(code)),
        "connect_or_create": bool(RE_CONNECT.search(code)),
        "nested_writes": bool(RE_NESTED_WRITE.search(code)),
        "transactions": bool(RE_TRANSACTION.search(code)),
        "group_by": bool(RE_GROUPBY.search(code)),
        "cursor_pagination": bool(RE_CURSOR.search(code)),
        "distinct": bool(RE_DISTINCT.search(code)),
        "raw_sql": bool(RE_RAW.search(code)),
        "order_by": bool(RE_ORDERBY.search(code)),
        "where": bool(RE_WHERE.search(code)),
    }


def extract_call_sites(rel_path: str, code: str) -> tuple[list[CallSite], dict[str, bool]]:
    """Return (call sites, feature flags) for one TypeScript source file."""
    sites: list[CallSite] = []
    for m in RE_CALL.finditer(code):
        _client, model_hint, op = m.group(1), m.group(2), m.group(3)
        line = code.count("\n", 0, m.start()) + 1
        if op in CRUD_OPS:
            family = "crud"
        elif op in AGG_OPS:
            family = "group_by" if op == "groupBy" else "aggregation"
        elif op.startswith("$"):
            family = "raw_sql" if "Raw" in op else "transaction"
        else:
            continue
        start = max(0, code.rfind("\n", 0, m.start()))
        end = code.find("\n", m.end())
        snippet = code[start : end if end != -1 else None].strip()[:240]
        sites.append(
            CallSite(
                op=op, model_hint=model_hint.lower(), family=family, file=rel_path, line=line, snippet=snippet
            )
        )
    if RE_TRANSACTION.search(code):
        for m in RE_TRANSACTION.finditer(code):
            line = code.count("\n", 0, m.start()) + 1
            start = max(0, code.rfind("\n", 0, m.start()))
            end = code.find("\n", m.end())
            sites.append(
                CallSite(
                    op="$transaction",
                    model_hint="",
                    family="transaction",
                    file=rel_path,
                    line=line,
                    snippet=code[start : end if end != -1 else None].strip()[:240],
                )
            )
    return sites, _families(code)


def classify_layer(rel_path: str, code: str) -> list[str]:
    """Which architectural layer(s) does this file live in? (report Part 2 extract list)"""
    layers: list[str] = []
    p = rel_path.lower()
    if RE_CLASS_REPO.search(code) or ".repository." in p or ".repo." in p or "/repositories/" in p:
        layers.append("repository")
    if RE_SERVICE.search(code) or ".service." in p or "/services/" in p:
        layers.append("service")
    if RE_ROUTER.search(code) or "/routes/" in p or "/routers/" in p or ".controller." in p:
        layers.append("api")
    if RE_GRAPHQL.search(code) or "resolvers/" in p:
        layers.append("graphql")
    return layers


def extract_from_snapshot(snapshot_dir: Path) -> list[ExtractionRecord]:
    """Extract all (schema, callsites) pairs from datasets/raw/<slug>/."""
    meta_path = snapshot_dir / "meta.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files_dir = snapshot_dir / "files"

    ts_files: dict[str, str] = {}
    for p in sorted(files_dir.glob("*.ts")):
        rel = p.name.replace("__", "/")[:-3]
        ts_files[rel] = p.read_text(encoding="utf-8", errors="replace")

    migrations: list[str] = []
    mig_dir = files_dir / "migrations"
    if mig_dir.exists():
        for p in sorted(mig_dir.glob("*.sql")):
            migrations.append(p.read_text(encoding="utf-8", errors="replace"))

    records: list[ExtractionRecord] = []
    for p in sorted(files_dir.glob("*.prisma")):
        schema_text = p.read_text(encoding="utf-8", errors="replace")
        parsed = parse_schema(schema_text)
        if not parsed.models:
            continue  # broken schema -> quality filter territory, not extraction

        rec = ExtractionRecord(
            repo=meta["full_name"],
            track=meta.get("track", "mined"),
            stack=meta.get("stack", "library"),
            license=meta.get("live_license_spdx", "UNKNOWN"),
            snapshot_ref=meta.get("snapshot_ref", ""),
            schema_path=p.name.replace("__", "/"),
            schema=schema_text,
            schema_hash=sha256_text(schema_text),
            models=[m.name for m in parsed.models],
            migrations=migrations,
        )
        rel_root = str(p.name.replace("__", "/")[:-7]).rsplit("/", 1)[0]
        for rel, code in ts_files.items():
            sites, feats = extract_call_sites(rel, code)
            rec.call_sites.extend(sites)
            for k, v in feats.items():
                rec.features[k] = rec.features.get(k, False) or v
            # only attribute layers to files plausibly inside the prisma app
            if rel_root in rel or rel_root == ".":
                for layer in classify_layer(rel, code):
                    if rel not in rec.layers[layer]:
                        rec.layers[layer].append(rel)
        records.append(rec)
    return records


def extract_all(raw_root: Path) -> list[ExtractionRecord]:
    out: list[ExtractionRecord] = []
    for snap in sorted(raw_root.iterdir()):
        if snap.is_dir():
            try:
                out.extend(extract_from_snapshot(snap))
            except Exception:  # noqa: BLE001
                log.exception("extraction failed for %s", snap)
    log.info("extracted %d schema records from %s", len(out), raw_root)
    return out
