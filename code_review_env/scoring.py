"""Shared, deterministic scoring primitives; source snippets are never executed."""
from __future__ import annotations

import re


def classification_metrics(flagged, bugs, files) -> dict:
    """File-level metrics; a correctly cleared all-safe patch has F1 = 1."""
    flagged, bugs, files = set(flagged), set(bugs), set(files)
    tp, fp, fn = len(flagged & bugs), len(flagged - bugs), len(bugs - flagged)
    tn = len(files - flagged - bugs)
    precision = tp / (tp + fp) if tp + fp else float(not bugs)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1)


def report_quality(report: str, session=None) -> float:
    """Report structure proxy, not a verification of semantic correctness.

    With session context, require the actual CVE and a file that was read.
    Token/word count alone never earns credit.
    """
    text = (report or "").lower()
    if not text.strip():
        return 0.0
    cve = getattr(session, "episode", {}).get("cve_id", "").lower() if session else ""
    score = 0.4 if (cve in text if cve else bool(re.search(r"\bcve-\d{4}-\d{4,}\b", text))) else 0.0
    if re.search(r"\b(buffer|overflow|injection|traversal|rce|xss|privilege|race condition|use-after-free|authentication|bypass)\b", text):
        score += 0.3
    has_detail = bool(re.search(r"\b(line\s+\d+|function\s+\w+|variable\s+\w+|call\s+\w+)", text))
    if session is not None:
        reads = getattr(session, "reads", set())
        has_detail = has_detail and any(path.lower() in text for path in reads)
    if has_detail:
        score += 0.3
    return min(score, 1.0)
