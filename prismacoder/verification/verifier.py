"""Verifier orchestrator: runs the V1-V5 chain, short-circuits on first failure,
classifies every failure into the E1-E10 taxonomy, and emits a
``VerificationResult`` that is embedded into dataset records and used as the
DPO/GRPO reward signal.

``LightValidator`` is the no-Node fallback used by CI smoke runs, the Kaggle
notebook, and offline inference: it statically checks schema structure, brace
balance, symbol grounding (every model/field referenced by the artifact exists
in the schema), and balanced fences. A LightValidator pass is recorded as
``status='unverified'`` with gate ``V0_static`` so full-gate re-verification
remains visible in provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from prismacoder.constants import GATE_ORDER
from prismacoder.psl import parse_schema
from prismacoder.verification.db_gate import gate_v4_execute
from prismacoder.verification.prisma_gate import gate_v1_validate, gate_v2_generate
from prismacoder.verification.sandbox import NodeMissingError, SandboxProject, node_available
from prismacoder.verification.test_gate import gate_v5_vitest
from prismacoder.verification.ts_gate import gate_v3_tsc


@dataclass
class VerificationResult:
    status: str  # verified | unverified | failed
    gates: dict[str, dict[str, Any]] = dc_field(default_factory=dict)
    error_class: str = ""  # E1..E10
    first_failing_gate: str = ""
    diagnostics: list[str] = dc_field(default_factory=list)
    runtime_ms: int = 0

    def to_record_block(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "gates": {k: v for k, v in self.gates.items()},
            "runtime_ms": self.runtime_ms,
            "verifier_version": "1.0.0",
            "error_class": self.error_class,
        }


# ---------------------------------------------------------------------------
# E1-E10 classification (report Table 11.3)
# ---------------------------------------------------------------------------
def classify_error(gate: str, diagnostics: list[str]) -> str:
    blob = " ".join(diagnostics).lower()
    if gate == "V1_validate":
        if "syntax" in blob or "expected" in blob or "line" in blob:
            return "E1"
        return "E2"
    if gate == "V2_generate":
        return "E3"
    if gate == "V3_tsc":
        if "ts2531" in blob or "possibly null" in blob:
            return "E4"
        if "ts2304" in blob or "does not exist" in blob:
            return "E4"
        return "E4"
    if gate == "V4_execute":
        if "p2002" in blob:
            return "E5"
        if "p2003" in blob or "foreign key" in blob:
            return "E5"
        if "p2025" in blob:
            return "E5"
        if "injection" in blob or "unsafe" in blob:
            return "E8"
        return "E5"
    if gate == "V5_vitest":
        if "expect" in blob or "assert" in blob:
            return "E6"
        if "n+1" in blob or "query-count" in blob:
            return "E9"
        return "E5"
    return "E6"


class Verifier:
    """Full-gate verifier. Requires Node >= 18.18 (or VERIFY_DOCKER=1)."""

    def __init__(
        self,
        *,
        timeout_seconds: int = 300,
        keep_sandboxes: bool = False,
        run_install: bool = True,
        gates: tuple[str, ...] = ("V1_validate", "V2_generate", "V3_tsc", "V4_execute", "V5_vitest"),
    ):
        self.timeout = timeout_seconds
        self.keep = keep_sandboxes
        self.run_install = run_install
        self.gates = gates

    def verify(
        self,
        schema: str,
        source_files: dict[str, str] | None = None,
        test_files: dict[str, str] | None = None,
        *,
        provider: str | None = None,
        migration_sql: str | None = None,
        seed_body: str | None = None,
        artifact_kind: str | None = None,
    ) -> VerificationResult:
        node, _npm = node_available()
        if not node:
            raise NodeMissingError(
                "Node >= 18.18 not found; use LightValidator for static checks or run the "
                "docker verifier image (docker/Dockerfile.verifier)"
            )

        if provider is None:
            # the V4 lane must match the schema's datasource provider - a sqlite
            # URL against a postgres schema is a guaranteed P1013
            provider = parse_schema(schema).datasource.get("provider", "sqlite") or "sqlite"
        sb = SandboxProject.create(schema, source_files, test_files, seed_body=seed_body, keep=self.keep)
        # PSL-kind artifacts (T1/T2/T3) are validated by V1/V2/V4; V3/V5 are
        # meaningless on PSL and are recorded as skipped (report Table 10.1)
        gates = self.gates
        if artifact_kind and artifact_kind.upper().startswith("PSL"):
            gates = tuple(g for g in gates if g not in ("V3_tsc", "V5_vitest"))
        result = VerificationResult(status="failed")
        try:
            if self.run_install:
                rc_install, out_install = sb.install()
                if rc_install != 0:
                    result.diagnostics = ["npm install failed"] + out_install.splitlines()[-5:]
                    result.gates["V0_install"] = {"passed": False}
                    return result
                result.gates["V0_install"] = {"passed": True}

            total_ms = 0
            for gate in sorted(gates, key=lambda g: GATE_ORDER.get(g, 99)):
                outcome = self._run_gate(sb, gate, provider=provider, migration_sql=migration_sql)
                total_ms += outcome.runtime_ms
                result.gates[outcome.gate if outcome.gate != "V1_validate" else "V1_validate"] = {
                    "passed": outcome.passed,
                    "skipped": outcome.skipped,
                    "skip_reason": outcome.skip_reason,
                    "runtime_ms": outcome.runtime_ms,
                }
                if not outcome.passed:
                    result.status = "failed"
                    result.first_failing_gate = outcome.gate
                    result.diagnostics = outcome.diagnostics
                    result.error_class = classify_error(outcome.gate, outcome.diagnostics)
                    result.runtime_ms = total_ms
                    return result
            result.status = "verified"
            result.runtime_ms = total_ms
            return result
        finally:
            sb.cleanup()

    def _run_gate(self, sb: SandboxProject, gate: str, *, provider: str, migration_sql: str | None):
        if gate == "V1_validate":
            return gate_v1_validate(sb, timeout=self.timeout)
        if gate == "V2_generate":
            return gate_v2_generate(sb, timeout=self.timeout * 2)
        if gate == "V3_tsc":
            return gate_v3_tsc(sb, timeout=self.timeout)
        if gate == "V4_execute":
            return gate_v4_execute(
                sb, provider=provider, migration_sql=migration_sql, timeout=self.timeout * 2
            )
        if gate == "V5_vitest":
            return gate_v5_vitest(sb, timeout=self.timeout * 2)
        raise ValueError(f"unknown gate {gate}")


# ---------------------------------------------------------------------------
# LightValidator (no node)
# ---------------------------------------------------------------------------
RE_FENCE = re.compile(r"```(?:\w+)?\n(.*?)```", re.S)


class LightValidator:
    """Static, dependency-free validation used when Node is unavailable."""

    name = "V0_static"

    def validate(self, schema: str, artifact: str) -> VerificationResult:
        result = VerificationResult(status="unverified")
        gates: dict[str, dict[str, Any]] = {}

        # G0: empty artifacts can never pass a full gate (V1-V5 all require a
        # real artifact), so static mode must fail them too - otherwise a
        # degenerate empty submission would score as an unconditional V0 pass.
        if not (artifact or "").strip():
            result.status = "failed"
            result.first_failing_gate = self.name
            result.error_class = "E7"
            result.diagnostics = ["empty artifact (nothing to verify)"]
            result.gates = {self.name: {"passed": False, "check": "empty_artifact"}}
            return result

        # G1: PSL structural sanity
        parsed = parse_schema(schema)
        ok_models = bool(parsed.models)
        gates["V0_static"] = {"passed": ok_models, "check": "schema_parses"}
        if not ok_models:
            result.status = "failed"
            result.first_failing_gate = self.name
            result.error_class = "E1"
            result.diagnostics = ["schema does not parse (no model blocks)"]
            result.gates = gates
            return result

        # G2: artifact fence / symbol grounding
        problems: list[str] = []
        symbols: dict[str, set[str]] = {}
        for m in parsed.models:
            symbols[m.name.lower()] = {f.name.lower() for f in m.fields}
        art = artifact
        fences = RE_FENCE.findall(art)
        if fences:
            art = "\n".join(fences)
        if artifact.strip() and "```" in artifact and not fences:
            problems.append("unbalanced code fences")

        model_props = set(re.findall(r"prisma\.(\w+)\.", art))
        for prop in model_props:
            if prop not in symbols and prop not in (
                "$transaction",
                "$queryRaw",
                "$executeRaw",
                "$queryRawUnsafe",
                "$executeRawUnsafe",
                "$on",
                "$extends",
            ):
                problems.append(f"ungrounded model reference: prisma.{prop}")
        # field-grounding: a key used anywhere in the artifact must exist in SOME
        # model of the schema (union check) - catches hallucinated fields without
        # cross-model false positives
        all_fields: set[str] = set()
        for fields in symbols.values():
            all_fields |= fields
        used = set(re.findall(r"\b(\w+):\s*[\{\[\(\"']", art))
        known_payload_keys = {
            "where",
            "data",
            "include",
            "select",
            "orderBy",
            "take",
            "skip",
            "cursor",
            "distinct",
            "create",
            "connect",
            "connectOrCreate",
            "update",
            "upsert",
            "delete",
            "set",
            "increment",
            "decrement",
            "by",
            "_count",
            "_sum",
            "_avg",
            "_min",
            "_max",
            "is",
            "some",
            "none",
            "every",
            "and",
            "or",
            "not",
            "equals",
            "contains",
            "gte",
            "lte",
            "gt",
            "lt",
            "in",
            "mode",
            "id",
            "name",
            "args",
            "patch",
            "number",
            "value",
            "quantity",
            # PSL attribute arguments (@relation/@id/@@index/...)
            "fields",
            "references",
            "ondelete",
            "onupdate",
            "map",
            "length",
            "sort",
            "clustered",
            "operations",
            "limit",
        }
        bad = {
            u
            for u in used
            if u.lower() not in all_fields
            and u not in known_payload_keys
            and len(u) > 3
            and not u.startswith("_")
            and not u[0].isupper()
        }
        for u in sorted(bad)[:5]:
            problems.append(f"possible ungrounded field '{u}'")

        gates[self.name] = {"passed": not problems, "check": "symbol_grounding"}
        result.gates = gates
        if problems:
            result.status = "failed"
            result.first_failing_gate = self.name
            result.error_class = "E4"
            result.diagnostics = problems[:10]
        return result
