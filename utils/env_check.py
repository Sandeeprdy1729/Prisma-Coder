#!/usr/bin/env python3
"""Environment sanity check + repo tree printer.

python utils/env_check.py            # check toolchain availability
python utils/env_check.py --tree     # print the repository layout
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "MISS"
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="store_true")
    args = ap.parse_args(argv)

    if args.tree:
        skip = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "runs",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
        }
        for p in sorted(REPO_ROOT.rglob("*")):
            if any(part in skip for part in p.parts):
                continue
            rel = p.relative_to(REPO_ROOT)
            depth = len(rel.parts) - 1
            print("  " * depth + ("|- " if p.is_dir() else "|  " * depth + "") + rel.name)
        return 0

    from prismacoder import __version__

    print(f"PrismaCoder {__version__}\n")

    ok = True
    py = sys.version_info
    ok &= check("python", py >= (3, 10), f"{py.major}.{py.minor}.{py.micro}")
    node = shutil.which("node")
    if check("node (V1-V5 gates)", node is not None, node or "static-only fallback"):
        out = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        check("node version >= 18.18", out.lstrip("v") >= "18.18", out)
    check("npm", shutil.which("npm") is not None, shutil.which("npm") or "")
    check("git", shutil.which("git") is not None, shutil.which("git") or "")

    for mod, extra in (
        ("yaml", "core"),
        ("requests", "core"),
        ("tqdm", "core"),
        ("numpy", "eval"),
        ("matplotlib", "eval"),
        ("torch", "training"),
        ("transformers", "training"),
        ("peft", "training"),
        ("trl", "training"),
        ("fastapi", "serve"),
        ("gradio", "serve"),
        ("huggingface_hub", "hub"),
    ):
        try:
            __import__(mod)
            check(f"python:{mod}", True, extra)
        except ImportError:
            check(f"python:{mod}", False, f"pip install -e '.[{extra}]'")

    check("GITHUB_TOKEN", bool(__import__("os").environ.get("GITHUB_TOKEN")), "dataset crawl")
    check("HF_TOKEN", bool(__import__("os").environ.get("HF_TOKEN")), "hub release")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
