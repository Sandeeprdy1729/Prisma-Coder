"""Gate V5: vitest behavioral suite."""

from __future__ import annotations

import time

from prismacoder.verification.prisma_gate import GateOutcome


def gate_v5_vitest(sb, timeout: int = 600) -> GateOutcome:
    started = time.monotonic()
    rc, out = sb.npm(["test", "--silent"], timeout=timeout)
    report = sb.read_vitest_report()
    diagnostics: list[str] = []
    if rc != 0:
        for ln in out.splitlines():
            low = ln.lower()
            if ("fail" in low or "error" in low or "assert" in low) and len(ln.strip()) < 400:
                diagnostics.append(ln.strip())
        diagnostics = diagnostics[:15] or ["vitest suite failed (no parseable diagnostics)"]
    passed = rc == 0
    if report is not None:
        totals = report.get("numPassedTests"), report.get("numFailedTests")
        if totals[0] is not None:
            passed = (totals[1] or 0) == 0 and totals[0] > 0
            if not passed:
                diagnostics = [f"vitest: {totals[1]} failed / {totals[0]} passed"]
    return GateOutcome("V5_vitest", passed, rc, diagnostics, int((time.monotonic() - started) * 1000))
