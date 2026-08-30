"""Secret detection (gitleaks-equivalent rule set, dependency-free).

Every rule is (compiled regex, severity, secret kind). Matches are reported
with file/line and the matched text is redacted. Records with HIGH severity
findings are dropped outright; MEDIUM findings are redacted and kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from prismacoder.io_utils import write_jsonl

RE_HEX64 = r"\b[a-f0-9]{64}\b"
RE_B64_40 = r"\b[A-Za-z0-9+/]{40,}\b"


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    severity: str  # high | medium
    kind: str
    redact: bool = True


RULES: tuple[Rule, ...] = (
    Rule(re.compile(r"A3T[A-Z0-9]{16}|AKIA[0-9A-Z]{16}"), "high", "aws_access_key_id"),
    Rule(re.compile(r"(?i)aws(.{0,20})?['\"][0-9a-zA-Z/+]{40}['\"]"), "high", "aws_secret_key"),
    Rule(re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH |PGP |)PRIVATE KEY-----"), "high", "private_key"),
    Rule(re.compile(r"ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82}"), "high", "github_pat"),
    Rule(re.compile(r"gho_[A-Za-z0-9]{36}|ghu_[A-Za-z0-9]{36}|ghs_[A-Za-z0-9]{36}"), "high", "github_oauth"),
    Rule(re.compile(r"sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}"), "high", "openai_key"),
    Rule(re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "high", "google_api_key"),
    Rule(re.compile(r"(?i)sg\.[0-9a-fA-F]{40}"), "high", "sendgrid_key"),
    Rule(re.compile(r"sk_live_[0-9a-zA-Z]{24}"), "high", "stripe_live_key"),
    Rule(re.compile(r"rk_live_[0-9a-zA-Z]{24}"), "high", "stripe_live_rk"),
    Rule(re.compile(r"xox[baprs]-[0-9A-Za-z\-]{10,}"), "high", "slack_token"),
    Rule(
        re.compile(
            r"(?i)(api[_-]?key|apikey|secret|token|passwd|password)\s*[:=]\s*['\"]"
            r"(?!($|%|\{\{|<|\\))"
            r"[A-Za-z0-9_\-]{16,}['\"]"
        ),
        "medium",
        "generic_credential",
    ),
    Rule(re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\."), "high", "jwt"),
    Rule(re.compile(r"postgres(?:ql)?://[^:\s\"']+:[^@\s\"']+@"), "medium", "database_url"),
    Rule(re.compile(r"mongodb(?:\+srv)?://[^:\s\"']+:[^@\s\"']+@"), "medium", "database_url"),
    Rule(re.compile(r"mysql://[^:\s\"']+:[^@\s\"']+@"), "medium", "database_url"),
    Rule(re.compile(RE_HEX64), "medium", "hex64_possible_secret"),
    Rule(re.compile(RE_B64_40), "low", "base64_blob"),
)


@dataclass
class Finding:
    kind: str
    severity: str
    line: int
    matched: str  # redacted form
    raw: str  # full match (never written to disk)


def redact(match: str, keep: int = 4) -> str:
    """Redact a secret but keep a 4-char prefix for diagnostics."""
    return match[:keep] + "..." + f"<redacted:{len(match)}>"


def scan_text(text: str) -> list[Finding]:
    findings: list[Finding] = []
    seen_spans: set[tuple[int, int]] = set()
    for rule in RULES:
        for m in rule.pattern.finditer(text):
            span = m.span()
            if any(s <= span[0] and span[1] <= e for s, e in seen_spans):
                continue  # covered by a higher-priority match
            seen_spans.add(span)
            line = text.count("\n", 0, span[0]) + 1
            findings.append(Finding(rule.kind, rule.severity, line, redact(m.group(0)), m.group(0)))
    return findings


def scan_and_clean(text: str) -> tuple[str, list[Finding], bool]:
    """Returns (cleaned_text, findings, drop_record).

    HIGH-severity findings force a drop (the secret may be load-bearing for
    the sample); MEDIUM/LOW findings are replaced in-place with a placeholder
    constant that keeps line structure identical.
    """
    findings = scan_text(text)
    if any(f.severity == "high" for f in findings):
        return text, findings, True
    cleaned = text
    for f in findings:
        if f.severity in ("medium", "low"):
            cleaned = cleaned.replace(f.raw, "REDACTED_SECRET_PLACEHOLDER")
    return cleaned, findings, False


def scan_directory_tree(files: dict[str, str]) -> tuple[dict[str, str], list[dict]]:
    """Scan a {relpath: content} mapping. Returns (cleaned_files, report_rows)."""
    cleaned: dict[str, str] = {}
    report: list[dict] = []
    for path, content in files.items():
        new, findings, drop = scan_and_clean(content)
        for f in findings:
            report.append(
                {
                    "file": path,
                    "kind": f.kind,
                    "severity": f.severity,
                    "line": f.line,
                    "matched": f.matched,
                    "dropped": drop,
                }
            )
        if not drop:
            cleaned[path] = new
    return cleaned, report


def write_secret_report(report: list[dict], path: str) -> int:
    return write_jsonl(path, report)
