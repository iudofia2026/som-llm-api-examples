#!/usr/bin/env python3
"""Audit saved OpenAI-compatible JSON responses for truncation and token use.

Point this at a directory of JSON sidecars from a bulk job. It recursively scans
`*.json` files and reports:

- files whose first choice finished with `finish_reason == "length"`;
- completion token counts when `usage.completion_tokens` is present;
- files that could not be parsed as JSON.

Example:

    ./09_audit_outputs.py ./outputs
    ./09_audit_outputs.py ./outputs --warn-over 8192
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class ResponseAudit:
    path: Path
    finish_reason: str | None
    completion_tokens: int | None


@dataclass(frozen=True)
class AuditSummary:
    parsed_count: int
    invalid_count: int
    length_count: int
    token_counts: list[int]
    over_warn_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing JSON sidecars")
    parser.add_argument(
        "--warn-over",
        type=int,
        default=8192,
        help="Count responses with completion_tokens above this value",
    )
    parser.add_argument(
        "--show-length-files",
        action="store_true",
        help="Print paths for files that finished because of max token length",
    )
    return parser.parse_args()


def iter_json_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    return value if isinstance(value, dict) else None


def first_choice_finish_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    finish_reason = first_choice.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def completion_tokens(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    value = usage.get("completion_tokens")
    return value if isinstance(value, int) and value >= 0 else None


def audit_file(path: Path) -> ResponseAudit | None:
    payload = load_json(path)
    if payload is None:
        return None

    return ResponseAudit(
        path=path,
        finish_reason=first_choice_finish_reason(payload),
        completion_tokens=completion_tokens(payload),
    )


def summarize(audits: list[ResponseAudit], invalid_count: int, warn_over: int) -> AuditSummary:
    token_counts = [audit.completion_tokens for audit in audits if audit.completion_tokens is not None]
    length_count = sum(1 for audit in audits if audit.finish_reason == "length")
    over_warn_count = sum(1 for value in token_counts if value > warn_over)

    return AuditSummary(
        parsed_count=len(audits),
        invalid_count=invalid_count,
        length_count=length_count,
        token_counts=token_counts,
        over_warn_count=over_warn_count,
    )


def percentile(sorted_values: list[int], pct: float) -> int | None:
    if not sorted_values:
        return None

    index = round((len(sorted_values) - 1) * pct)
    return sorted_values[index]


def print_summary(summary: AuditSummary, warn_over: int) -> None:
    total_files = summary.parsed_count + summary.invalid_count
    print(f"JSON files:       {total_files}")
    print(f"Parsed responses: {summary.parsed_count}")
    print(f"Invalid JSON:     {summary.invalid_count}")
    print(f"finish=length:    {summary.length_count}")
    print(f">{warn_over} tokens:   {summary.over_warn_count}")

    if not summary.token_counts:
        print("completion_tokens: not present")
        return

    sorted_counts = sorted(summary.token_counts)
    print("completion_tokens:")
    print(f"  count: {len(sorted_counts)}")
    print(f"  min:   {sorted_counts[0]}")
    print(f"  p50:   {int(median(sorted_counts))}")
    print(f"  p90:   {percentile(sorted_counts, 0.90)}")
    print(f"  p95:   {percentile(sorted_counts, 0.95)}")
    print(f"  max:   {sorted_counts[-1]}")


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise SystemExit(f"Path does not exist: {args.root}")

    audits: list[ResponseAudit] = []
    invalid_count = 0

    for path in iter_json_files(args.root):
        audit = audit_file(path)
        if audit is None:
            invalid_count += 1
            continue
        audits.append(audit)

    summary = summarize(audits, invalid_count, args.warn_over)
    print_summary(summary, args.warn_over)

    if args.show_length_files:
        length_files = [audit.path for audit in audits if audit.finish_reason == "length"]
        if length_files:
            print("\nFiles with finish_reason=length:")
            for path in length_files:
                print(path)


if __name__ == "__main__":
    main()
