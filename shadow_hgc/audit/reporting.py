from __future__ import annotations

from collections.abc import Iterable

from shadow_hgc.audit.config_checks import parse_metric


def row_is_valid_for_best(row: dict) -> bool:
    if row.get("status", "completed") != "completed":
        return False
    if row.get("valid_for_best") is False:
        return False
    return parse_metric(row.get("accuracy")) is not None


def best_rows_by_dataset(rows: Iterable[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for row in rows:
        if not row_is_valid_for_best(row):
            continue
        dataset = str(row.get("dataset", ""))
        if not dataset:
            continue
        acc = parse_metric(row.get("accuracy"))
        if acc is None:
            continue
        current = best.get(dataset)
        if current is None or acc > float(current["_accuracy_for_sort"]):
            copy = dict(row)
            copy["_accuracy_for_sort"] = acc
            best[dataset] = copy
    for row in best.values():
        row.pop("_accuracy_for_sort", None)
    return best


def markdown_table(rows: list[dict], fields: list[str]) -> list[str]:
    if not rows:
        return ["None."]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines
