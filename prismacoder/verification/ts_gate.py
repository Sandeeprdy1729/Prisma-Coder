"""Gate V3: TypeScript compilation (tsc --noEmit)."""

from __future__ import annotations

import time

from prismacoder.verification.prisma_gate import GateOutcome, parse_ts_errors


def gate_v3_tsc(sb, timeout: int = 300) -> GateOutcome:
    started = time.monotonic()
    rc, out = sb.npx(["tsc", "--noEmit"], timeout=timeout)
    return GateOutcome("V3_tsc", rc == 0, rc, parse_ts_errors(out), int((time.monotonic() - started) * 1000))
