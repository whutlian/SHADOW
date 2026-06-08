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
    "budget_mode",
    "ratio",
    "ratio_base",
    "target_budget",
    "max_target_budget",
    "budget_rounding",
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
    "shadow_ratio_target_target",
    "shadow_ratio_non_target",
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
ALLOWED_LOSS_TYPES = {"weighted", "unweighted", "clipped", "class_balanced", "sqrt_weighted", "sqrt_weighted_logit_adjusted"}
ALLOWED_MODELS = {"relation_linear", "relation_mlp"}
ALLOWED_BUDGET_MODES = {"ratio", "count"}
ALLOWED_RATIO_BASES = {"train_target", "all_target"}
ALLOWED_BUDGET_ROUNDING = {"nearest", "ceil", "floor"}
ALLOWED_SHADOW_POLICIES = {"fixed", "rank_adaptive"}
ALLOWED_FEATURE_MODES = {"base", "diffusion", "metapath", "diffusion_metapath"}
ALLOWED_SKELETON_POLICIES = {"fixed_k", "coverage"}


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
    budget_mode = config.get("budget_mode", "count")
    if budget_mode not in ALLOWED_BUDGET_MODES:
        raise ValueError(f"budget_mode must be one of {sorted(ALLOWED_BUDGET_MODES)}")
    ratio_base = config.get("ratio_base", "train_target")
    if ratio_base not in ALLOWED_RATIO_BASES:
        raise ValueError(f"ratio_base must be one of {sorted(ALLOWED_RATIO_BASES)}")
    budget_rounding = config.get("budget_rounding", "nearest")
    if budget_rounding not in ALLOWED_BUDGET_ROUNDING:
        raise ValueError(f"budget_rounding must be one of {sorted(ALLOWED_BUDGET_ROUNDING)}")
    if budget_mode == "ratio" and float(config.get("ratio", 0.0)) <= 0.0:
        raise ValueError("ratio must be positive when budget_mode=ratio")
    target_budget = config.get("target_budget")
    if budget_mode == "count" and (target_budget is None or int(target_budget) <= 0):
        raise ValueError("target_budget must be positive when budget_mode=count")
    if int(config.get("k_s", 0)) < 0:
        raise ValueError("k_s must be non-negative")
    if int(config.get("shadow_assignment_b", 1)) != 1:
        raise ValueError("shadow_assignment_b must be 1 for the main path")
    shadow_policy = config.get("shadow_policy", "fixed")
    if shadow_policy not in ALLOWED_SHADOW_POLICIES:
        raise ValueError(f"shadow_policy must be one of {sorted(ALLOWED_SHADOW_POLICIES)}")
    feature_mode = config.get("feature_mode", "base")
    if feature_mode not in ALLOWED_FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {sorted(ALLOWED_FEATURE_MODES)}")
    skeleton_policy = config.get("skeleton_policy", "fixed_k")
    if skeleton_policy not in ALLOWED_SKELETON_POLICIES:
        raise ValueError(f"skeleton_policy must be one of {sorted(ALLOWED_SKELETON_POLICIES)}")
    if float(config.get("degree_scale", 0.0)) < 0.0:
        raise ValueError("degree_scale must be non-negative")
    if float(config.get("signature_degree_eta", 0.0)) < 0.0:
        raise ValueError("signature_degree_eta must be non-negative")
    if int(config.get("min_proto_per_class", 1)) < 1:
        raise ValueError("min_proto_per_class must be positive")
