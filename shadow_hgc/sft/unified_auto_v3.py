from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shadow_hgc.sft.domain_transport import DomainTransportConfig, compute_domain_transport_strength
from shadow_hgc.sft.unified_auto_v2 import (
    T40Schedule,
    apply_candidate_policy as apply_candidate_policy_v2,
    compute_t40_schedule,
    policy_selection_score as policy_selection_score_v2_equivalent,
    schedule_to_row_fields_v2,
)


PUBLIC_METHOD_NAME = "Shadow-HGC-STT-U"
FIXED_CANDIDATE_POLICIES_V3: tuple[str, ...] = (
    "auto_base",
    "coverage_heavy",
    "domain_coverage",
    "teacher_transport",
    "high_fidelity",
    "domain_transport",
)


@dataclass(frozen=True)
class T41Schedule(T40Schedule):
    public_method_name: str = PUBLIC_METHOD_NAME
    domain_transport_strength: float = 0.0
    domain_transport_active: bool = False


def _from_t40(base: T40Schedule, *, strength: float, active: bool) -> T41Schedule:
    return T41Schedule(
        policy_name=base.policy_name,
        class_capacity_b=base.class_capacity_b,
        budget_phase=base.budget_phase,
        teacher_reliability_q=base.teacher_reliability_q,
        domain_gap_train_all=base.domain_gap_train_all,
        selection_weights=dict(base.selection_weights),
        loss_weights=dict(base.loss_weights),
        soft_temperature=base.soft_temperature,
        student_family=base.student_family,
        student_internal_style=base.student_internal_style,
        hidden_dim=base.hidden_dim,
        epochs=base.epochs,
        teacher_cache=base.teacher_cache,
        public_method_name=PUBLIC_METHOD_NAME,
        domain_transport_strength=float(strength),
        domain_transport_active=bool(active),
    )


def compute_t41_schedule(
    *,
    condensed_nodes: int,
    num_classes: int,
    teacher_valid_acc: float | None,
    majority_valid_acc: float | None,
    domain_gap_train_all: float,
    num_nodes: int,
    num_teacher_nodes: int | None = None,
    is_ultra_dataset: bool = False,
    dense_cache_budget_bytes: int = 256 * 1024 * 1024,
    domain_transport_cfg: DomainTransportConfig | None = None,
) -> T41Schedule:
    base = compute_t40_schedule(
        condensed_nodes=int(condensed_nodes),
        num_classes=int(num_classes),
        teacher_valid_acc=teacher_valid_acc,
        majority_valid_acc=majority_valid_acc,
        domain_gap_train_all=float(domain_gap_train_all),
        num_nodes=int(num_nodes),
        num_teacher_nodes=num_teacher_nodes,
        is_ultra_dataset=bool(is_ultra_dataset),
        dense_cache_budget_bytes=int(dense_cache_budget_bytes),
    )
    cfg = domain_transport_cfg or DomainTransportConfig()
    strength = compute_domain_transport_strength(float(domain_gap_train_all), float(base.class_capacity_b), cfg)
    active = strength >= float(cfg.activation_threshold)
    return _from_t40(base, strength=strength, active=active)


def apply_candidate_policy(schedule: T41Schedule, policy: str) -> T41Schedule:
    policy = str(policy)
    if policy not in FIXED_CANDIDATE_POLICIES_V3:
        raise ValueError(f"unknown T41 candidate policy: {policy}")
    if policy == "domain_transport":
        base = apply_candidate_policy_v2(schedule, "domain_coverage")
        selection = dict(base.selection_weights)
        loss = dict(base.loss_weights)
        selection["domain"] = max(selection.get("domain", 0.0), schedule.selection_weights.get("domain", 0.0) * 1.25)
        loss["lambda_domain"] = min(0.30, max(float(loss.get("lambda_domain", 0.0)), float(schedule.loss_weights.get("lambda_domain", 0.0)) * 1.50))
        return T41Schedule(
            policy_name="domain_transport",
            class_capacity_b=base.class_capacity_b,
            budget_phase=base.budget_phase,
            teacher_reliability_q=base.teacher_reliability_q,
            domain_gap_train_all=base.domain_gap_train_all,
            selection_weights=selection,
            loss_weights=loss,
            soft_temperature=base.soft_temperature,
            student_family=base.student_family,
            student_internal_style=base.student_internal_style,
            hidden_dim=base.hidden_dim,
            epochs=base.epochs,
            teacher_cache=base.teacher_cache,
            public_method_name=PUBLIC_METHOD_NAME,
            domain_transport_strength=schedule.domain_transport_strength,
            domain_transport_active=schedule.domain_transport_active,
        )
    base = apply_candidate_policy_v2(schedule, policy)
    return T41Schedule(
        policy_name=base.policy_name,
        class_capacity_b=base.class_capacity_b,
        budget_phase=base.budget_phase,
        teacher_reliability_q=base.teacher_reliability_q,
        domain_gap_train_all=base.domain_gap_train_all,
        selection_weights=dict(base.selection_weights),
        loss_weights=dict(base.loss_weights),
        soft_temperature=base.soft_temperature,
        student_family=base.student_family,
        student_internal_style=base.student_internal_style,
        hidden_dim=base.hidden_dim,
        epochs=base.epochs,
        teacher_cache=base.teacher_cache,
        public_method_name=PUBLIC_METHOD_NAME,
        domain_transport_strength=schedule.domain_transport_strength,
        domain_transport_active=schedule.domain_transport_active,
    )


def policy_selection_score_v3(
    row: dict[str, Any],
    *,
    alpha_macro: float = 0.10,
    beta_prior: float = 0.04,
    gamma_domain: float = 0.04,
    delta_transport: float = 0.06,
    eta_overfit: float = 0.04,
) -> float:
    valid_acc = float(row.get("valid_acc", 0.0) or 0.0)
    macro = float(row.get("valid_macro_f1", 0.0) or 0.0)
    prior = float(row.get("selected_prior_kl", 0.0) or 0.0)
    selected_domain_gap = float(row.get("domain_gap_after", row.get("domain_coverage_gap", 0.0)) or 0.0)
    transport_gain = float(row.get("domain_transport_gain", 0.0) or 0.0)
    overfit = float(row.get("domain_overfit_proxy", 0.0) or 0.0)
    return float(
        valid_acc
        + float(alpha_macro) * macro
        - float(beta_prior) * prior
        - float(gamma_domain) * selected_domain_gap
        + float(delta_transport) * transport_gain
        - float(eta_overfit) * overfit
    )


def select_best_candidate(rows: list[dict[str, Any]], *, domain_transport_valid_guard: float = 5e-4) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate rows cannot be empty")
    scored: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["score_v2_equivalent"] = policy_selection_score_v2_equivalent(out)
        out["score_v3"] = policy_selection_score_v3(out)
        out["policy_selection_score"] = out["score_v3"]
        scored.append(out)
    best = max(scored, key=lambda item: (float(item["score_v3"]), float(item.get("valid_acc", 0.0) or 0.0), str(item.get("selected_policy", ""))))
    if str(best.get("selected_policy", "")) == "domain_transport":
        max_valid = max(float(item.get("valid_acc", 0.0) or 0.0) for item in scored)
        best_valid = float(best.get("valid_acc", 0.0) or 0.0)
        if best_valid < max_valid - float(domain_transport_valid_guard):
            eligible = [item for item in scored if float(item.get("valid_acc", 0.0) or 0.0) >= max_valid - float(domain_transport_valid_guard)]
            if eligible:
                guarded = max(
                    eligible,
                    key=lambda item: (float(item["score_v3"]), float(item.get("valid_acc", 0.0) or 0.0), str(item.get("selected_policy", ""))),
                )
                guarded["policy_selection_guard"] = "domain_transport_valid_guard"
                return guarded
    best["policy_selection_guard"] = ""
    return best


def schedule_to_row_fields_v3(schedule: T41Schedule) -> dict[str, Any]:
    fields = schedule_to_row_fields_v2(schedule)
    fields.update(
        {
            "domain_transport_strength": float(schedule.domain_transport_strength),
            "domain_transport_active": bool(schedule.domain_transport_active),
            "public_method_name": PUBLIC_METHOD_NAME,
        }
    )
    return fields
