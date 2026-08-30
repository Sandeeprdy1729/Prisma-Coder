"""Gate V4: schema -> database execution (migrate/db push + optional seed).

Lanes:
- ``sqlite``    (default): `prisma db push` against a local file DB - no services.
- ``postgres``  (opt-in via VERIFY_POSTGRES_URL): `prisma db push` against a
  running Postgres (docker compose provides one); skipped cleanly when the
  URL is absent or unreachable.
- raw-SQL migrations (T3 tasks) are dry-run against sqlite when provided.
"""

from __future__ import annotations

import os
import time
import urllib.parse

from prismacoder.verification.prisma_gate import GateOutcome, parse_prisma_errors


def _pg_reachable(url: str, timeout: float = 2.0) -> bool:
    """TCP-only reachability probe for the postgres lane (no driver needed)."""
    import socket

    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def gate_v4_execute(
    sb, *, provider: str = "sqlite", migration_sql: str | None = None, timeout: int = 600
) -> GateOutcome:
    started = time.monotonic()

    if provider in ("postgresql", "postgres"):
        url = os.environ.get("VERIFY_POSTGRES_URL", "")
        if not url or not _pg_reachable(url):
            return GateOutcome(
                "V4_execute",
                True,
                0,
                [],
                0,
                skipped=True,
                skip_reason="postgres lane unavailable (no reachable VERIFY_POSTGRES_URL)",
            )
        env = {"DATABASE_URL": url}
        rc, out = sb.npx(
            ["prisma", "db", "push", "--skip-generate", "--force-reset"], timeout=timeout, env=env
        )
        if rc != 0:
            return GateOutcome(
                "V4_execute", False, rc, parse_prisma_errors(out), int((time.monotonic() - started) * 1000)
            )
        return GateOutcome("V4_execute", True, 0, [], int((time.monotonic() - started) * 1000))

    if provider == "mysql":
        return GateOutcome(
            "V4_execute",
            True,
            0,
            [],
            0,
            skipped=True,
            skip_reason="mysql lane not provisioned in CI sandboxes",
        )

    # sqlite lane (default)
    rc, out = sb.npx(["prisma", "db", "push"], timeout=timeout)
    if rc != 0:
        return GateOutcome(
            "V4_execute", False, rc, parse_prisma_errors(out), int((time.monotonic() - started) * 1000)
        )

    if migration_sql:
        # dry-run the provided migration SQL statements against the sqlite file
        return _run_sqlite_statements(sb, migration_sql, started, timeout)

    return GateOutcome("V4_execute", True, 0, [], int((time.monotonic() - started) * 1000))


def _run_sqlite_statements(sb, sql: str, started: float, timeout: int) -> GateOutcome:
    """Execute migration statements via a tiny tsx runner against dev.db."""
    statements = [s.strip() + ";" for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]
    if not statements:
        return GateOutcome("V4_execute", True, 0, [], int((time.monotonic() - started) * 1000))
    runner = (
        'import Database from "better-sqlite3";\n'
        'import fs from "node:fs";\n'
        'const url = process.env.DATABASE_URL?.replace("file:", "") || "./prisma/dev.db";\n'
        "const db = new Database(url);\n"
        'const stmts = JSON.parse(process.argv[1] || "[]") as string[];\n'
        "for (const [i, s] of stmts.entries()) {\n"
        "  try { db.exec(s); } catch (e: any) {\n"
        "    console.error(`statement ${i} failed: ${e.message}`);\n"
        "    process.exit(1);\n"
        "  }\n"
        "}\n"
        "db.close();\n"
        "console.log(`executed ${stmts.length} statements`);\n"
    )
    (sb.root / "sqlite_runner.ts").write_text(runner, encoding="utf-8")
    import json as _json

    rc, out = sb.npx(["tsx", "sqlite_runner.ts", _json.dumps(statements)], timeout=timeout)
    return GateOutcome(
        "V4_execute",
        rc == 0,
        rc,
        [ln for ln in out.splitlines() if "failed" in ln][:10],
        int((time.monotonic() - started) * 1000),
    )
