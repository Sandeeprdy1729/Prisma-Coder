"""PII scrubbing (report 9.1: 'PII scrub').

Strategy for a *code corpus* (different from document PII):
- emails, phone numbers, person-looking display names, geo coordinates that
  appear in seed data: replaced with structural placeholders.
- OAuth/bearer strings are handled by ``prismacoder.secrets`` (higher priority).
- Placeholders are PSL/TS-safe so parsed artifacts remain valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,24}\b")
PHONE_RE = re.compile(
    r"(?<![\w.\-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?![\w.\-])"
)
COORD_RE = re.compile(r"\b(?:lat(?:itude)?|lng|lon(?:gitude)?)\s*[:=]\s*-?\d{1,3}\.\d{3,}\b", re.IGNORECASE)
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
TWITTER_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{4,15}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# examples/docs domains are intentionally kept - they carry no personal data
SAFE_EMAIL_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "test.test",
    "acme.com",
    "prisma.io",
    "gmail.com.test",
    "company.com",
    "email.com",
}


@dataclass
class PiiEdit:
    kind: str
    line: int
    placeholder: str


def _is_safe_email(addr: str) -> bool:
    domain = addr.split("@", 1)[1].lower()
    return domain in SAFE_EMAIL_DOMAINS


def scrub_text(text: str) -> tuple[str, list[PiiEdit]]:
    edits: list[PiiEdit] = []
    out = text

    def sub_email(m: re.Match[str]) -> str:
        if _is_safe_email(m.group(0)):
            return m.group(0)
        edits.append(PiiEdit("email", out.count("\n", 0, m.start()) + 1, "user@example.invalid"))
        return "user@example.invalid"

    def sub_phone(m: re.Match[str]) -> str:
        edits.append(PiiEdit("phone", out.count("\n", 0, m.start()) + 1, "5550100"))
        return "5550100"

    def sub_coord(m: re.Match[str]) -> str:
        edits.append(PiiEdit("coord", out.count("\n", 0, m.start()) + 1, "lat: 0.0"))
        return m.group(0).split(":")[0].split("=")[0].strip() + ": 0.0"

    def sub_ip(m: re.Match[str]) -> str:
        edits.append(PiiEdit("ipv4", out.count("\n", 0, m.start()) + 1, "127.0.0.1"))
        return "127.0.0.1"

    def sub_ssn(m: re.Match[str]) -> str:
        edits.append(PiiEdit("ssn", out.count("\n", 0, m.start()) + 1, "000-00-0000"))
        return "000-00-0000"

    out = EMAIL_RE.sub(sub_email, out)
    out = PHONE_RE.sub(sub_phone, out)
    out = COORD_RE.sub(sub_coord, out)
    out = IPV4_RE.sub(sub_ip, out)
    out = SSN_RE.sub(sub_ssn, out)
    return out, edits


def scrub_files(files: dict[str, str]) -> tuple[dict[str, str], list[dict]]:
    """Scrub a {relpath: content} mapping; returns cleaned files + edit log."""
    cleaned: dict[str, str] = {}
    log: list[dict] = []
    for path, content in files.items():
        new, edits = scrub_text(content)
        cleaned[path] = new
        for e in edits:
            log.append({"file": path, "kind": e.kind, "line": e.line, "placeholder": e.placeholder})
    return cleaned, log
