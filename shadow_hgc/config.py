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
    "model",
    "degree_scale",
    "signature_degree_eta",
    "min_proto_per_class",
    "budget_alpha",
    "shadow_budget_policy",
    "shadow_non_target_ratio",
    "shadow_target_target_ratio",
    "min_shadows_per_relation",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "strict_budget",
    "seed",
    "resource_budget",
}

ALLOWED_PROJECTION_TYPES = {"raw", "random"}
ALLOWED_LOSS_TYPES = {"weighted", "unweighted", "clipped", "class_balanced", "sqrt_weighted"}
ALLOWED_MODELS = {"relation_linear", "relation_mlp"}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS - set(config))
    if missing:
        raise ValueError(f"{path} is missing required config fields: {missing}")
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: dict[str, Any]) -> None:
    projection_type = config.get("projection_type")
    if projection_type not in ALLOWED_PROJECTION_TYPES:
        raise ValueError(f"projection_type must be one of {sorted(ALLOWED_PROJECTION_TYPES)}")
    loss_type = config.get("loss_type")
    if loss_type not in ALLOWED_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {sorted(ALLOWED_LOSS_TYPES)}")
    model = config.get("model", "relation_linear")
    if model not in ALLOWED_MODELS:
        raise ValueError(f"model must be one of {sorted(ALLOWED_MODELS)}")
    if int(config.get("k_s", 0)) < 0:
        raise ValueError("k_s must be non-negative")
    if int(config.get("shadow_assignment_b", 1)) != 1:
        raise ValueError("shadow_assignment_b must be 1 for the main path")
    if float(config.get("degree_scale", 0.0)) < 0.0:
        raise ValueError("degree_scale must be non-negative")
    if float(config.get("signature_degree_eta", 0.0)) < 0.0:
        raise ValueError("signature_degree_eta must be non-negative")
    if int(config.get("min_proto_per_class", 1)) < 1:
        raise ValueError("min_proto_per_class must be positive")
