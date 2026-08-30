"""PrismaForge GitHub crawler (Phase 1, mined track).

Pipeline per repository:
1. fetch repo metadata + license (live verification, not the seed expectation)
2. list the default-branch tree
3. pull schema.prisma files, migration SQL, and TypeScript sources
4. persist a *raw* snapshot per repo:  datasets/raw/{owner}__{repo}/
     - meta.json           repo metadata + provenance
     - tree.json           file listing
     - files/*.prisma      raw schemas
     - files/*.ts          raw TS sources (capped per repo)
     - files/migrations/*.sql

Design notes
------------
* Clone-less: everything goes through the REST API so the crawler works on
  ephemeral CI machines with no git binary and no GitHub app.
* Per-repo file caps keep the crawl bounded (default 400 TS files).
* ``--offline`` mode replays an existing raw snapshot (used by CI and tests).
* Raw snapshots are the immutable build input; every downstream stage reads
  them and records ``source.commit``-style provenance from meta.json.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import sys
import time
from pathlib import Path
from typing import Any

from prismacoder.github_api import GitHubClient
from prismacoder.io_utils import ensure_dir, write_json
from prismacoder.sources import SEARCH_QUERIES, SeedRepo, seed_repos

log = logging.getLogger("prismaforge.crawler")

TS_GLOBS = (
    "**/prisma/*.ts",
    "**/prisma/seed.ts",
    "**/*.service.ts",
    "**/*.repository.ts",
    "**/*.repo.ts",
    "**/routes/*.ts",
    "**/routers/*.ts",
    "**/resolvers/*.ts",
    "**/controllers/*.ts",
    "**/*.controller.ts",
    "**/queries/*.ts",
    "**/mutations/*.ts",
)
MAX_TS_FILES_PER_REPO = 400
MAX_FILE_BYTES = 300_000
DISCOVER_MAX_REPOS = 300


def _glob_match(path: str, patterns: tuple[str, ...]) -> bool:
    norm = path.lstrip("./")
    return any(
        fnmatch.fnmatch(norm, p.replace("**/", "*/").replace("/**", "/*")) or fnmatch.fnmatch(norm, p)
        for p in patterns
    )


def _slug(full_name: str) -> str:
    return full_name.replace("/", "__")


def crawl_repo(
    client: GitHubClient, repo: SeedRepo | dict[str, Any], out_root: Path, max_ts: int = MAX_TS_FILES_PER_REPO
) -> dict[str, Any] | None:
    """Crawl one repository into out_root/raw/<slug>/. Returns summary or None."""
    full = repo.full_name if isinstance(repo, SeedRepo) else repo["full_name"]
    meta = client.repo(full)
    if not meta:
        log.info("repo %s unavailable (404/private) - skipping", full)
        return None

    license_info = meta.get("license") or {}
    license_spdx = license_info.get("spdx_id") or "UNKNOWN"

    slug = _slug(full)
    repo_dir = ensure_dir(out_root / "raw" / slug)
    files_dir = ensure_dir(repo_dir / "files")
    mig_dir = ensure_dir(files_dir / "migrations")

    tree = client.default_branch_tree(full) or []
    write_json(repo_dir / "tree.json", tree)

    counts = {"schemas": 0, "ts": 0, "migrations": 0}
    for node in tree:
        if node.get("type") != "blob":
            continue
        path = node.get("path", "")
        size = int(node.get("size", 0))
        if size > MAX_FILE_BYTES:
            continue
        if path.endswith(".prisma"):
            content = client.blob(full, path)
            if content:
                target = files_dir / (path.replace("/", "__") + ".prisma")
                target.write_text(content, encoding="utf-8")
                counts["schemas"] += 1
        elif _glob_match(path, TS_GLOBS) and counts["ts"] < max_ts:
            content = client.blob(full, path)
            if content:
                target = files_dir / (path.replace("/", "__") + ".ts")
                target.write_text(content, encoding="utf-8")
                counts["ts"] += 1
        elif "/migrations/" in path and path.endswith("/migration.sql"):
            content = client.blob(full, path)
            if content:
                target = mig_dir / (path.replace("/", "__") + ".sql")
                target.write_text(content, encoding="utf-8")
                counts["migrations"] += 1

    meta_out = {
        "full_name": full,
        "track": repo.track if isinstance(repo, SeedRepo) else repo.get("track", "mined"),
        "stack": repo.stack if isinstance(repo, SeedRepo) else repo.get("stack", "library"),
        "expected_license": repo.expected_license if isinstance(repo, SeedRepo) else "",
        "live_license_spdx": license_spdx,
        "stars": meta.get("stargazers_count", 0),
        "default_branch": meta.get("default_branch", "main"),
        "snapshot_ref": meta.get("pushed_at") or meta.get("updated_at") or "",
        "description": meta.get("description") or "",
        "counts": counts,
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "crawler_version": "1.0.0",
    }
    write_json(repo_dir / "meta.json", meta_out)
    log.info("crawled %s: %s", full, counts)
    return meta_out


def discover_repos(client: GitHubClient, max_repos: int = DISCOVER_MAX_REPOS) -> list[str]:
    """Code-search expansion beyond the seed list. Returns 'owner/repo' names."""
    found: list[str] = []
    seen: set[str] = set()
    for q in SEARCH_QUERIES:
        for item in client.search_code(q, per_page=50, pages=2):
            full = (item.get("repository") or {}).get("full_name")
            if full and full not in seen:
                seen.add(full)
                found.append(full)
            if len(found) >= max_repos:
                return found
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PrismaForge GitHub crawler (mined track)")
    ap.add_argument("--out", default="datasets", help="output root (default: datasets)")
    ap.add_argument("--max-ts", type=int, default=MAX_TS_FILES_PER_REPO)
    ap.add_argument(
        "--discover",
        type=int,
        default=0,
        help="also crawl N repos discovered via code search (0 = seed list only)",
    )
    ap.add_argument("--seed-repos", default="", help="comma-separated owner/repo overrides")
    ap.add_argument("--no-ecosystem", action="store_true", help="official track only")
    ap.add_argument("--offline", action="store_true", help="verify snapshots exist; no network")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    out_root = ensure_dir(args.out)

    if args.offline:
        raw = out_root / "raw"
        snaps = sorted(p.name for p in raw.iterdir() if p.is_dir()) if raw.exists() else []
        log.info("offline mode: %d raw snapshots present under %s", len(snaps), raw)
        return 0

    client = GitHubClient()
    if not client.token:
        log.warning("GITHUB_TOKEN not set - anonymous rate limit is 60 req/h; a crawl will stall quickly")

    targets: list[Any] = []
    if args.seed_repos:
        for name in args.seed_repos.split(","):
            name = name.strip()
            if "/" in name:
                targets.append(SeedRepo(name, "mined", "UNKNOWN", "library", 5))
    else:
        targets = list(seed_repos(include_ecosystem=not args.no_ecosystem))

    if args.discover:
        extra = discover_repos(client, args.discover)
        log.info("code search discovered %d additional repos", len(extra))
        targets += [SeedRepo(n, "mined", "UNKNOWN", "library", 5) for n in extra]

    ok, fail = 0, 0
    for repo in targets:
        try:
            summary = crawl_repo(client, repo, out_root, max_ts=args.max_ts)
            ok += 1 if summary else 0
            fail += 0 if summary else 1
        except Exception:  # noqa: BLE001 - one bad repo must not kill the crawl
            log.exception("crawl failed for %s", repo)
            fail += 1
        time.sleep(0.5)  # polite pacing

    manifest = {
        "crawled_ok": ok,
        "crawled_failed": fail,
        "targets": len(targets),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out_root / "raw" / "crawl_manifest.json", manifest)
    log.info("crawl done: %s", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
