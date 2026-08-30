"""Gate V1 (prisma format + validate) and V2 (prisma generate).

Both gates shell out to the Prisma CLI inside a :class:`SandboxProject`.
Diagnostics are parsed into structured form (error code + message + line when
available) and mapped to the E1-E10 taxonomy by ``verifier.classify_error``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dc_field


@dataclass
class GateOutcome:
    gate: str
    passed: bool
    exit_code: int
    diagnostics: list[str] = dc_field(default_factory=list)
    runtime_ms: int = 0
    skipped: bool = False
    skip_reason: str = ""


RE_PSL_ERROR = re.compile(r"error:\s*(?P<msg>[^\n]+?)\s*\n\s*-->\s*(?P<file>[^:]+):(?P<line>\d+)", re.S)
RE_TS_ERROR = re.compile(r"(?P<file>[^\s(]+)\((?P<line>\d+),\d+\):\s*error TS(?P<code>\d+):\s*(?P<msg>.+)")


def parse_prisma_errors(output: str) -> list[str]:
    out = [f"{m.group('msg').strip()} (line {m.group('line')})" for m in RE_PSL_ERROR.finditer(output)]
    if not out and output.strip() and ("error" in output.lower()):
        tail = [ln.strip() for ln in output.splitlines() if "error" in ln.lower()][:5]
        out = tail
    return out


def parse_ts_errors(output: str) -> list[str]:
    return [
        f"TS{m.group('code')} {m.group('msg').strip()} ({m.group('file')}:{m.group('line')})"
        for m in RE_TS_ERROR.finditer(output)
    ]


def gate_v1_validate(sb, timeout: int = 300) -> GateOutcome:
    """prisma format && prisma validate."""
    import time

    started = time.monotonic()
    rc_fmt, out_fmt = sb.npx(["prisma", "format"], timeout=timeout)
    if rc_fmt != 0:
        return GateOutcome(
            "V1_validate",
            False,
            rc_fmt,
            parse_prisma_errors(out_fmt),
            int((time.monotonic() - started) * 1000),
        )
    rc_val, out_val = sb.npx(["prisma", "validate"], timeout=timeout)
    return GateOutcome(
        "V1_validate",
        rc_val == 0,
        rc_val,
        parse_prisma_errors(out_val),
        int((time.monotonic() - started) * 1000),
    )


def gate_v2_generate(sb, timeout: int = 600) -> GateOutcome:
    """prisma generate (codegen gate; requires network on a cold sandbox)."""
    import time

    started = time.monotonic()
    rc, out = sb.npx(["prisma", "generate"], timeout=timeout)
    return GateOutcome(
        "V2_generate", rc == 0, rc, parse_prisma_errors(out), int((time.monotonic() - started) * 1000)
    )
