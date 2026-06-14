from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from shadow_hgc.ultra.papers100m_t36_contract import DISCO_PARITY_RATIOS, percent_to_ratio_decimal


DISCO_BASELINE_ROWS: list[dict[str, Any]] = [
    {
        "dataset": "ogbn-papers100M",
        "backend": "SGC",
        "ratio_percent": 0.005,
        "ratio_decimal": 0.00005,
        "random_acc": 0.128,
        "herding_acc": 0.213,
        "kcenter_acc": 0.087,
        "disco_acc": 0.483,
        "whole_dataset_acc": 0.633,
        "source": "DisCo Table 4",
        "notes": "accuracies converted from percent",
    },
    {
        "dataset": "ogbn-papers100M",
        "backend": "SGC",
        "ratio_percent": 0.010,
        "ratio_decimal": 0.00010,
        "random_acc": 0.178,
        "herding_acc": 0.268,
        "kcenter_acc": 0.104,
        "disco_acc": 0.487,
        "whole_dataset_acc": 0.633,
        "source": "DisCo Table 4",
        "notes": "accuracies converted from percent",
    },
    {
        "dataset": "ogbn-papers100M",
        "backend": "SGC",
        "ratio_percent": 0.020,
        "ratio_decimal": 0.00020,
        "random_acc": 0.298,
        "herding_acc": 0.362,
        "kcenter_acc": 0.172,
        "disco_acc": 0.496,
        "whole_dataset_acc": 0.633,
        "source": "DisCo Table 4",
        "notes": "accuracies converted from percent",
    },
    {
        "dataset": "ogbn-papers100M",
        "backend": "SGC",
        "ratio_percent": 0.050,
        "ratio_decimal": 0.00050,
        "random_acc": 0.375,
        "herding_acc": 0.437,
        "kcenter_acc": 0.263,
        "disco_acc": 0.509,
        "whole_dataset_acc": 0.633,
        "source": "DisCo Table 4",
        "notes": "accuracies converted from percent",
    },
]


DISCO_FIELDS = [
    "dataset",
    "backend",
    "ratio_percent",
    "ratio_decimal",
    "random_acc",
    "herding_acc",
    "kcenter_acc",
    "disco_acc",
    "whole_dataset_acc",
    "source",
    "notes",
]


def ensure_disco_baseline_csv(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DISCO_FIELDS)
        writer.writeheader()
        for row in DISCO_BASELINE_ROWS:
            writer.writerow(row)
    return target


def load_disco_baseline(path: str | Path) -> dict[float, dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        ensure_disco_baseline_csv(target)
    rows: dict[float, dict[str, Any]] = {}
    with target.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in ("ratio_percent", "ratio_decimal", "random_acc", "herding_acc", "kcenter_acc", "disco_acc", "whole_dataset_acc"):
                row[key] = float(row[key])
            ratio = float(row["ratio_decimal"])
            rows[ratio] = row
    validate_disco_baseline(rows)
    return rows


def validate_disco_baseline(rows: dict[float, dict[str, Any]]) -> None:
    missing = [ratio for ratio in DISCO_PARITY_RATIOS if ratio not in rows]
    if missing:
        raise ValueError(f"missing DisCo parity ratios: {missing}")
    for ratio, row in rows.items():
        expected = percent_to_ratio_decimal(row["ratio_percent"])
        if abs(expected - float(row["ratio_decimal"])) > 5e-12:
            raise ValueError(f"percent/decimal mismatch for ratio {ratio}: {row}")
        for key in ("random_acc", "herding_acc", "kcenter_acc", "disco_acc", "whole_dataset_acc"):
            value = float(row[key])
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{key} must be stored as fraction, got {value}")


def attach_disco_metrics(row: dict[str, Any], baseline: dict[float, dict[str, Any]]) -> dict[str, Any]:
    out = dict(row)
    ratio = float(out.get("requested_full_node_ratio", out.get("ratio_decimal", 0.0)) or 0.0)
    base = baseline.get(ratio)
    if base is None:
        return out
    accuracy = out.get("accuracy", "")
    out["disco_acc"] = float(base["disco_acc"])
    out["random_acc_baseline"] = float(base["random_acc"])
    out["herding_acc_baseline"] = float(base["herding_acc"])
    out["kcenter_acc_baseline"] = float(base["kcenter_acc"])
    if accuracy != "":
        acc = float(accuracy)
        disco = float(base["disco_acc"])
        out["absolute_gain_vs_disco"] = acc - disco
        out["relative_error_reduction_vs_disco"] = (acc - disco) / max(1.0 - disco, 1e-12)
        out["beats_disco"] = acc >= disco
        out["beats_random"] = acc >= float(base["random_acc"])
        out["beats_herding"] = acc >= float(base["herding_acc"])
        out["beats_kcenter"] = acc >= float(base["kcenter_acc"])
    return out


def parity_allowed(*, backend: str, ratio: float, denominator: int) -> bool:
    return str(backend).lower() == "sgc" and float(ratio) in DISCO_PARITY_RATIOS and int(denominator) == 111_059_956
