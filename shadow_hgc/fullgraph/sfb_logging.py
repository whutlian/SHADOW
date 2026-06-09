from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], *, fieldnames: list[str] | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in materialized:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines
