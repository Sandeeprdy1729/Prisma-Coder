"""Curated seed repository list for the PrismaForge mined track (report 9.1).

Tracks:
- ``official``: prisma org + stack templates - golden seeds / eval anchors.
- ``ecosystem``: high-signal open-source SaaS / stacks that use Prisma heavily.

License expectations are annotated but ALWAYS re-verified at crawl time via the
GitHub API (``prismacoder.license_filter``) - the expectation is only used for
prioritization. Repos whose live license is copyleft are skipped automatically
(the ``expected_license`` below records what we expect, which doubles as a
self-test of the license gate).

GitHub topic / search expansion beyond the seed list is handled by
``SEARCH_QUERIES`` at the bottom of this file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedRepo:
    full_name: str  # "owner/repo"
    track: str  # official | ecosystem
    expected_license: str  # verified again at crawl time
    stack: str  # nextjs | nestjs | express | redwood | blitz | graphql | trpc | library | docs
    priority: int  # 1 = crawl first
    notes: str = ""


OFFICIAL_REPOS: tuple[SeedRepo, ...] = (
    SeedRepo(
        "prisma/prisma-examples",
        "official",
        "Apache-2.0",
        "library",
        1,
        "canonical examples across providers/frameworks",
    ),
    SeedRepo(
        "prisma/docs", "official", "Apache-2.0", "docs", 1, "docs track for CPT; guides contain client idioms"
    ),
    SeedRepo("prisma/website", "official", "Apache-2.0", "docs", 2, "blog + guide sources"),
    SeedRepo(
        "prisma/prisma",
        "official",
        "Apache-2.0",
        "library",
        2,
        "orm repo; functional-test schemas are gold seeds",
    ),
    SeedRepo(
        "t3-oss/create-t3-app",
        "official",
        "MIT",
        "nextjs",
        1,
        "T3 stack scaffolding with Prisma+Next.js templates",
    ),
    SeedRepo(
        "redwoodjs/redwood",
        "official",
        "MIT",
        "redwood",
        1,
        "services layer = repository/service patterns over Prisma",
    ),
    SeedRepo("blitz-js/blitz", "official", "MIT", "blitz", 2, "queries/mutations over Prisma"),
    SeedRepo("trpc/trpc", "official", "MIT", "trpc", 2, "routers consuming Prisma in examples"),
    SeedRepo(
        "amplication/amplication",
        "official",
        "Apache-2.0",
        "nestjs",
        2,
        "generates Prisma services; rich CRUD corpus",
    ),
    SeedRepo("zenstackhq/zenstack", "official", "Apache-2.0", "library", 2, "schema-level policy samples"),
    SeedRepo("wasp-lang/wasp", "official", "MIT", "library", 3, "app specs compiling to Prisma"),
)

ECOSYSTEM_REPOS: tuple[SeedRepo, ...] = (
    SeedRepo("infisical/infisical", "ecosystem", "MIT", "express", 3, "API + services over Prisma"),
    SeedRepo("nextauthjs/next-auth", "ecosystem", "ISC", "library", 3, "adapter examples over Prisma"),
    SeedRepo("marmelab/react-admin", "ecosystem", "MIT", "library", 4, "data-provider examples"),
    SeedRepo("colinhacks/zod", "ecosystem", "MIT", "library", 4, "validation pairs for Prisma payloads"),
    SeedRepo("steven-tey/novel", "ecosystem", "Apache-2.0", "nextjs", 4, "editor + prisma usage"),
    SeedRepo(
        "boxyhq/jackson", "ecosystem", "Apache-2.0", "library", 4, "SSO service with Prisma store adapter"
    ),
)

ALL_SEED_REPOS: tuple[SeedRepo, ...] = OFFICIAL_REPOS + ECOSYSTEM_REPOS


def normalize_seed(repo: SeedRepo) -> SeedRepo | None:
    """Validate a seed entry; returns None when malformed (loader contract)."""
    parts = repo.full_name.split("/")
    if len(parts) != 2 or not all(parts):
        return None
    if repo.track not in ("official", "ecosystem"):
        return None
    return repo


def seed_repos(include_ecosystem: bool = True) -> list[SeedRepo]:
    """Curated crawl list, ordered by priority. Malformed entries are dropped."""
    pool = ALL_SEED_REPOS if include_ecosystem else OFFICIAL_REPOS
    kept = [r for r in (normalize_seed(r) for r in pool) if r is not None]
    return sorted(kept, key=lambda r: (r.priority, r.full_name))


# GitHub code-search queries used to DISCOVER additional repos beyond the seed list.
SEARCH_QUERIES: tuple[str, ...] = (
    "path:**/schema.prisma language:TypeScript stars:>100",
    "path:**/schema.prisma sqlite language:TypeScript stars:>50",
    "prisma $transaction language:TypeScript stars:>200",
    "prisma groupBy language:TypeScript stars:>100",
    "prisma $queryRaw language:TypeScript stars:>100",
    "nestjs prisma service language:TypeScript stars:>100",
    "nextjs prisma trpc language:TypeScript stars:>300",
)

# File globs fetched per repo during extraction.
TARGET_GLOBS: tuple[str, ...] = (
    "**/schema.prisma",
    "**/prisma/migrations/**/*.sql",
    "**/migrations/migration_lock.toml",
)
