"""Verification gates V1-V5 (report 7.9 / Table 10.1).

    V1  prisma validate (+format)   PSL semantics
    V2  prisma generate             codegen gate
    V3  tsc --noEmit                type gate
    V4  migrate + seed sandbox      execution gate (sqlite default; postgres lane)
    V5  vitest suite                behavioral gate

The gates run inside a Node sandbox (see ``sandbox.SandboxProject``), either on
the host (Node >= 18.18 on PATH) or inside the ``prismacoder/verifier`` docker
image (``VERIFY_DOCKER=1``). Gates short-circuit at the first failure and
return structured diagnostics that power the E1-E10 classifier and the
DPO/GRPO reward machinery.
"""

from prismacoder.verification.sandbox import NodeMissingError, SandboxProject
from prismacoder.verification.verifier import LightValidator, VerificationResult, Verifier

__all__ = [
    "SandboxProject",
    "NodeMissingError",
    "VerificationResult",
    "Verifier",
    "LightValidator",
]
