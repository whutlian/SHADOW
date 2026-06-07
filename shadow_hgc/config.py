from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_FIELDS = {
    "method",
    "dataset",
    "target_type",
    "directed_relations",
    "feature_dim",
    "projection_type",
    "standardization_scope",
    "M_tau",
    "M_r",
    "k_s",
    "shadow_assignment_b",
    "loss_type",
    "seed",
    "resource_budget",
}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(config))
    if missing:
        raise ValueError(f"{path} is missing required config fields: {missing}")
    return config
