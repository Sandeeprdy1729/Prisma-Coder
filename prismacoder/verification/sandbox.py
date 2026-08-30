"""Node sandbox: materializes a self-contained project for the V1-V5 gates.

Layout created under a temp dir (or a configured workspace):

    sandbox/
      package.json          pinned toolchain (prisma CLI, ts, vitest, tsx)
      tsconfig.json         strict TS config used by V3
      vitest.config.ts
      prisma/schema.prisma  the schema under test
      src/<name>.ts         artifacts under test
      tests/<name>.test.ts  V5 suite
      prisma/seed.ts        V4 seed (optional)

Everything is deterministic; dependencies install once into ``node_modules``
and are cached across runs (``PRISMACODER_SANDBOX_CACHE``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from prismacoder.constants import TOOLCHAIN

PACKAGE_JSON = {
    "name": "prismacoder-sandbox",
    "private": True,
    "version": "0.0.0",
    "prisma": {"schema": "prisma/schema.prisma"},
    "scripts": {
        "test": "vitest run --reporter=json --outputFile.vitest.json=__vitest__.json",
    },
    "devDependencies": {
        "typescript": TOOLCHAIN["typescript"],
        "prisma": TOOLCHAIN["prisma"],
        "@prisma/client": TOOLCHAIN["@prisma/client"],
        "vitest": TOOLCHAIN["vitest"],
        "tsx": TOOLCHAIN["tsx"],
        "@types/node": "^22.10.0",
    },
}

TSCONFIG = {
    "compilerOptions": {
        "target": "ES2022",
        "module": "ESNext",
        "moduleResolution": "bundler",
        "lib": ["ES2022"],
        "strict": True,
        "noEmit": True,
        "esModuleInterop": True,
        "skipLibCheck": True,
        "resolveJsonModule": True,
        "forceConsistentCasingInFileNames": True,
    },
    "include": ["src/**/*.ts", "tests/**/*.ts", "prisma/**/*.ts"],
        "types": ["node"],
}

VITEST_CONFIG = """import { defineConfig } from "vitest/config";
export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    testTimeout: 30000,
  },
});
"""

# Prisma 7 removed the datasource `url` from schema files and automatic .env
# loading; the CLI reads the connection URL from this config file instead.
PRISMA_CONFIG_TS = """import path from "node:path";
import fs from "node:fs";
import { defineConfig, env } from "prisma/config";

// Minimal .env loader (Prisma 7 no longer auto-loads .env files).
const envPath = path.join(process.cwd(), ".env");
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, "utf8").split("\\n")) {
    const m = line.match(/^\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*"?([^"\\n]*)"?.*$/);
    if (m && process.env[m[1]] === undefined) process.env[m[1]] = m[2];
  }
}

export default defineConfig({
  schema: path.join(process.cwd(), "prisma", "schema.prisma"),
  datasource: {
    url: env("DATABASE_URL"),
  },
});
"""

SEED_TEMPLATE = """import { PrismaClient } from "../app/generated/prisma";
const prisma = new PrismaClient();

async function main() {{
{body}
}}

main()
  .then(() => prisma.$disconnect())
  .catch(async (e) => {{
    console.error(e);
    await prisma.$disconnect();
    process.exit(1);
  }});
"""


class NodeMissingError(RuntimeError):
    """Raised when Node.js >= 18.18 is required but not found on PATH."""


def node_available(node_bin: str | None = None, npm_bin: str | None = None) -> tuple[str | None, str | None]:
    node = node_bin or os.environ.get("NODE_BIN") or shutil.which("node")
    npm = npm_bin or os.environ.get("NPM_BIN") or shutil.which("npm")
    if node:
        try:
            out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10)
            major, minor = out.stdout.strip().lstrip("v").split(".")[:2]
            if (int(major), int(minor)) < (18, 18):
                return None, npm
        except Exception:  # noqa: BLE001
            return None, npm
    return node, npm


@dataclass
class SandboxProject:
    root: Path
    keep: bool = False

    # -- creation --------------------------------------------------------------
    @classmethod
    def create(
        cls,
        schema: str,
        source_files: dict[str, str] | None = None,
        test_files: dict[str, str] | None = None,
        seed_body: str | None = None,
        env_url: str | None = None,
        keep: bool = False,
        base_dir: str | None = None,
    ) -> SandboxProject:
        base = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="pcsandbox-"))
        (base / "prisma").mkdir(parents=True, exist_ok=True)
        (base / "src").mkdir(exist_ok=True)
        (base / "tests").mkdir(exist_ok=True)

        (base / "package.json").write_text(json.dumps(PACKAGE_JSON, indent=2), encoding="utf-8")
        (base / "tsconfig.json").write_text(json.dumps(TSCONFIG, indent=2), encoding="utf-8")
        (base / "vitest.config.ts").write_text(VITEST_CONFIG, encoding="utf-8")
        (base / "prisma.config.ts").write_text(PRISMA_CONFIG_TS, encoding="utf-8")
        (base / "prisma" / "schema.prisma").write_text(schema, encoding="utf-8")

        for name, content in (source_files or {}).items():
            target = base / "src" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        for name, content in (test_files or {}).items():
            target = base / "tests" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        if seed_body:
            (base / "prisma" / "seed.ts").write_text(SEED_TEMPLATE.format(body=seed_body), encoding="utf-8")

        (base / ".env").write_text(f'DATABASE_URL="{env_url or "file:./dev.db"}"\n', encoding="utf-8")
        return cls(root=base, keep=keep)

    # -- command execution --------------------------------------------------------
    def run(
        self, cmd: list[str], timeout: int = 300, env_extra: dict[str, str] | None = None
    ) -> tuple[int, str]:
        env = os.environ.copy()
        env["CI"] = "1"
        env.setdefault("PRISMA_HIDE_UPDATE_MESSAGE", "1")
        if env_extra:
            env.update(env_extra)
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.root), capture_output=True, text=True, timeout=timeout, env=env
            )
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            return proc.returncode, output[-20000:]
        except subprocess.TimeoutExpired:
            return 124, f"TIMEOUT after {timeout}s: {' '.join(cmd)}"
        except FileNotFoundError:
            return 127, f"binary not found: {cmd[0]}"

    def npm(
        self, args: list[str], timeout: int = 600, env_extra: dict[str, str] | None = None
    ) -> tuple[int, str]:
        node, npm = node_available()
        if not node or not npm:
            raise NodeMissingError("Node.js >= 18.18 and npm are required for the verification gates")
        return self.run([npm, *args], timeout=timeout, env_extra=env_extra)

    def npx(
        self, args: list[str], timeout: int = 300, env_extra: dict[str, str] | None = None
    ) -> tuple[int, str]:
        node, npm = node_available()
        if not node or not npm:
            raise NodeMissingError("Node.js >= 18.18 and npm are required for the verification gates")
        npx = str(Path(npm).with_name("npx")) if Path(npm).name == "npm" else "npx"
        return self.run([npx, *args], timeout=timeout, env_extra=env_extra)

    def install(self) -> tuple[int, str]:
        """npm install (uses `npm ci`-like idempotent install of the pinned set)."""
        return self.npm(["install", "--no-audit", "--no-fund", "--loglevel", "error"], timeout=900)

    def read_vitest_report(self) -> dict | None:
        p = self.root / "__vitest__.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def cleanup(self) -> None:
        if not self.keep and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
