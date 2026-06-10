from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from shadow_hgc.ratio.scale_bucket import validate_t24_promoted_row


def read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    target.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    return target


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def ensure_report(path: str | Path, lines: list[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def fvalue(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bvalue(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def promotion_status(row: dict[str, Any], *, passed_gate: bool) -> tuple[str, str]:
    safety = validate_t24_promoted_row(row)
    if not safety["valid"]:
        return "blocked_forbidden", ",".join(safety["forbidden_flags"])
    if not passed_gate:
        return "not_promoted", "acceptance_gate_not_met"
    return "promoted", "passed_gate"
