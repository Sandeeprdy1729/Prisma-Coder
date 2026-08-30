"""Auto-repair loop (report Part 11 / Part 14).

Feeds gate diagnostics back into the model context and re-generates until the
artifact goes gate-green or the attempt budget is exhausted. Each attempt
records: gate verdicts, error class (E1-E10), and the diagnostic text.

Repair prompt shape:
    user  : original request + schema
    assist: failing artifact
    user  : "Your output failed <gate>: <diagnostics>. Return the corrected
             artifact only."
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Callable

from prismacoder.constants import ERROR_TAXONOMY
from prismacoder.verification.verifier import LightValidator, VerificationResult

GenerateFn = Callable[[list[dict[str, str]]], str]  # messages -> artifact text


@dataclass
class RepairAttempt:
    attempt: int
    artifact: str
    status: str  # verified | failed | unverified
    error_class: str
    diagnostics: list[str] = dc_field(default_factory=list)
    gate: str = ""


@dataclass
class RepairResult:
    artifact: str
    status: str
    attempts: list[RepairAttempt] = dc_field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == "verified"

    @property
    def error_summary(self) -> dict[str, Any]:
        last = self.attempts[-1] if self.attempts else None
        return {
            "attempts": len(self.attempts),
            "final_status": self.status,
            "final_gate": last.gate if last else "",
            "final_error_class": last.error_class if last else "",
            "taxonomy_meaning": ERROR_TAXONOMY.get(last.error_class, "") if last else "",
        }


class RepairLoop:
    def __init__(
        self, validator: LightValidator | None = None, max_attempts: int = 3, full_verifier: Any | None = None
    ):
        self.validator = validator or LightValidator()
        self.max_attempts = max_attempts
        self.full_verifier = full_verifier  # optional Verifier (Node gates)

    def run(
        self, instruction: str, schema: str, generate: GenerateFn, first_artifact: str | None = None
    ) -> RepairResult:
        messages: list[dict[str, str]] = [
            {"role": "user", "content": self._compose_request(instruction, schema)},
        ]
        result = RepairResult(artifact="", status="failed")
        artifact = first_artifact
        for attempt in range(1, self.max_attempts + 1):
            if artifact is None:
                artifact = generate(messages)
            verdict = self._verify(schema, artifact)
            result.attempts.append(
                RepairAttempt(
                    attempt=attempt,
                    artifact=artifact,
                    status=verdict.status,
                    error_class=verdict.error_class,
                    diagnostics=verdict.diagnostics,
                    gate=verdict.first_failing_gate or "V0_static",
                )
            )
            result.artifact = artifact
            result.status = "verified" if verdict.status != "failed" else "failed"
            if verdict.status != "failed":
                return result
            if attempt < self.max_attempts:
                messages = messages[:1] + [
                    {"role": "assistant", "content": artifact},
                    {"role": "user", "content": self._compose_repair(verdict)},
                ]
            artifact = None
        return result

    # -- helpers ------------------------------------------------------------------
    def _verify(self, schema: str, artifact: str) -> VerificationResult:
        if self.full_verifier is not None:
            try:
                return self.full_verifier.verify(schema, {"artifact.ts": artifact})
            except Exception:  # noqa: BLE001 - fall back to static on sandbox failure
                pass
        return self.validator.validate(schema, artifact)

    @staticmethod
    def _compose_request(instruction: str, schema: str) -> str:
        parts = [instruction]
        if schema.strip():
            parts.append(f"Here is the current schema.prisma:\n\n```prisma\n{schema.strip()}\n```")
        return "\n\n".join(parts)

    @staticmethod
    def _compose_repair(verdict: VerificationResult) -> str:
        diags = "\n".join(f"- {d}" for d in verdict.diagnostics[:8]) or "- (no diagnostics captured)"
        gate = verdict.first_failing_gate or "static validation"
        return (
            f"Your output failed {gate} (error class {verdict.error_class or 'unclassified'}):\n"
            f"{diags}\n\n"
            "Return the corrected, complete artifact only - no explanations, no restating "
            "of the request. Keep every part of the request that already passed."
        )
