"""Thin, dependency-light GitHub REST client with rate-limit awareness.

Endpoints used by the crawler:
- ``GET /search/code``        discover repos containing schema.prisma
- ``GET /search/repositories``
- ``GET /repos/{owner}/{repo}``            metadata + license
- ``GET /repos/{owner}/{repo}/git/trees/{sha}?recursive=1``  file listing
- ``GET /repos/{owner}/{repo}/contents/{path}``              blob fetch (base64)

The client never raises on 403/404 - it returns ``None`` so the crawler can
skip-and-continue; rate-limit sleeps are handled centrally.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

API = "https://api.github.com"
RATE_LIMIT_SLEEP = 2.0  # seconds to sleep on secondary rate limit
MAX_RETRIES = 3


class GitHubClient:
    def __init__(self, token: str | None = None, session: requests.Session | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "PrismaForge-crawler/1.0",
            }
        )
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    # -- core request -------------------------------------------------------
    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any] | None:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                log.warning("network error %s (attempt %d): %s", url, attempt, e)
                time.sleep(RATE_LIMIT_SLEEP * attempt)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (403, 429):
                reset = resp.headers.get("X-RateLimit-Reset")
                wait = RATE_LIMIT_SLEEP
                if reset:
                    wait = max(RATE_LIMIT_SLEEP, min(120.0, float(reset) - time.time() + 1))
                log.info("rate-limited on %s; sleeping %.1fs", url, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            log.warning("GET %s -> %s (attempt %d)", url, resp.status_code, attempt)
            time.sleep(RATE_LIMIT_SLEEP * attempt)
        return None

    # -- endpoints ----------------------------------------------------------
    def repo(self, full_name: str) -> dict[str, Any] | None:
        return self._get(f"{API}/repos/{full_name}")

    def default_branch_tree(self, full_name: str) -> list[dict[str, Any]] | None:
        meta = self.repo(full_name)
        if not meta:
            return None
        branch = meta.get("default_branch", "main")
        data = self._get(f"{API}/repos/{full_name}/git/trees/{branch}", {"recursive": "1"})
        if not data or data.get("truncated"):
            # truncated trees still yield most files; we take what we can get
            if not data:
                return None
        return data.get("tree", []) or []

    def blob(self, full_name: str, path: str, ref: str | None = None) -> str | None:
        params = {"ref": ref} if ref else None
        data = self._get(f"{API}/repos/{full_name}/contents/{path}", params)
        if not data or data.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - malformed base64 guard
            return None

    def search_code(self, query: str, per_page: int = 50, pages: int = 1) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in range(1, pages + 1):
            data = self._get(f"{API}/search/code", {"q": query, "per_page": per_page, "page": page})
            if not data:
                break
            items = data.get("items", [])
            out.extend(items)
            if len(items) < per_page:
                break
            time.sleep(RATE_LIMIT_SLEEP)  # search API: strict secondary limits
        return out

    def search_repos(self, query: str, per_page: int = 50, pages: int = 1) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in range(1, pages + 1):
            data = self._get(f"{API}/search/repositories", {"q": query, "per_page": per_page, "page": page})
            if not data:
                break
            items = data.get("items", [])
            out.extend(items)
            if len(items) < per_page:
                break
            time.sleep(RATE_LIMIT_SLEEP)
        return out

    def rate_limit(self) -> dict[str, Any]:
        data = self._get(f"{API}/rate_limit")
        return (data or {}).get("resources", {}).get("core", {})


def commit_sha_of(repo_meta: dict[str, Any]) -> str:
    """Best-effort snapshot id for provenance metadata."""
    return str(repo_meta.get("pushed_at") or repo_meta.get("updated_at") or "unknown")
