"""License gate (report 9.1): permissive-only ingress, copyleft excluded by policy.

The gate evaluates the *live* SPDX id returned by the GitHub API at crawl time
(``meta.json: live_license_spdx``), never the seed expectation. Repos with
``UNKNOWN``/``NONE`` license are rejected - "no license" is not "permissive".

Blocked families: GPL-*, AGPL-*, LGPL-*, SSPL, EUPL, and unlicensed content.
License text checks (for files that embed their own LICENSE) are supported via
``license_text_spdx``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from prismacoder.constants import ALLOWED_LICENSES, DENIED_LICENSES

# SPDX identifiers that normalize to an allowed family
_ALIASES = {
    "Apache-2.0": {"Apache-2.0", "Apache-2", "Apache 2.0", "Apache2"},
    "MIT": {"MIT"},
    "BSD-2-Clause": {"BSD-2-Clause", "BSD-2", "BSD 2-Clause", "BSD-2-Clause-Views"},
    "BSD-3-Clause": {"BSD-3-Clause", "BSD-3", "BSD 3-Clause"},
    "ISC": {"ISC"},
}


@dataclass(frozen=True)
class LicenseDecision:
    spdx: str
    allowed: bool
    reason: str


def normalize_spdx(spdx: str) -> str:
    spdx = (spdx or "").strip()
    for canonical, variants in _ALIASES.items():
        if spdx in variants:
            return canonical
    return spdx


def license_text_spdx(text: str) -> str:
    """Detect the SPDX family from a LICENSE file's body (fallback path)."""
    t = text[:4096].lower()
    if "apache license" in t and "version 2.0" in t:
        return "Apache-2.0"
    if "gnu affero general public license" in t:
        return "AGPL-3.0"
    if "gnu general public license" in t:
        m = re.search(r"version (\d)", t)
        return f"GPL-{m.group(1)}.0" if m else "GPL-3.0"
    if "gnu lesser general public license" in t:
        return "LGPL-3.0"
    if "permission is hereby granted, free of charge" in t and "mit" in t:
        return "MIT"
    if "isc license" in t or ("permission to use, copy, modify" in t and "isc" in t):
        return "ISC"
    if "redistribution and use in source and binary forms" in t:
        return "BSD-3-Clause"
    return "UNKNOWN"


def check_license(spdx_live: str, *, spdx_text: str = "") -> LicenseDecision:
    spdx = normalize_spdx(spdx_live) if spdx_live else normalize_spdx(spdx_text)
    if spdx in ("", "UNKNOWN", "NOASSERTION", "NONE", "UNLICENSED"):
        return LicenseDecision(spdx or "UNKNOWN", False, "no/unknown license: not permissive by default")
    if (
        any(spdx.startswith(d.split("-")[0]) and d in spdx for d in DENIED_LICENSES)
        or spdx in DENIED_LICENSES
    ):
        return LicenseDecision(spdx, False, f"copyleft/protected family {spdx} excluded by policy")
    if spdx in ALLOWED_LICENSES:
        return LicenseDecision(spdx, True, "permissive allowlist")
    return LicenseDecision(spdx, False, f"spdx {spdx} not on the permissive allowlist")


def filter_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Apply the gate to extraction/dataset records carrying ``license`` keys."""
    kept, dropped = [], []
    for r in records:
        dec = check_license(r.get("license", "UNKNOWN"))
        if dec.allowed:
            r["license"] = dec.spdx
            kept.append(r)
        else:
            dropped.append((r.get("repo", r.get("id", "?")), dec.reason))
    return kept, dropped
