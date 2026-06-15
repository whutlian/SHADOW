from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from shadow_hgc.sft.unified_schedule import DEFAULT_TEACHER_CACHE_DENSE_BUDGET_BYTES, TeacherCacheDecision


EPS = 1e-12


@dataclass(frozen=True)
class T40Schedule:
    policy_name: str
    class_capacity_b: float
    budget_phase: float
    teacher_reliability_q: float
    domain_gap_train_all: float
    selection_weights: dict[str, float]
    loss_weights: dict[str, float]
    soft_temperature: float
    student_family: str
    student_internal_style: str
    hidden_dim: int
    epochs: int
    teacher_cache: TeacherCacheDecision


def clip(value: float, low: float, high: float) -> float:
    return min(float(high), max(float(low), float(value)))


def capacity_phase(condensed_nodes: int, num_classes: int) -> tuple[float, float]:
    b = float(int(condensed_nodes)) / float(max(int(num_classes), 1))
    safe_b = max(b, EPS)
    tau = clip((math.log2(safe_b) - math.log2(16.0)) / (math.log2(256.0) - math.log2(16.0)), 0.0, 1.0)
    return b, tau


def teacher_reliability(
    teacher_valid_acc: float | None,
    majority_valid_acc: float | None,
    *,
    eps: float = EPS,
) -> float:
    if teacher_valid_acc is None or majority_valid_acc is None:
        return 0.0
    return clip((float(teacher_valid_acc) - float(majority_valid_acc)) / max(1.0 - float(majority_valid_acc), eps), 0.0, 1.0)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _normalize_selection(raw: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in raw.values())
    if total <= 0.0:
        return {key: 0.0 for key in raw}
    return {key: max(0.0, float(value)) / total for key, value in raw.items()}


def _topk_cache_bytes(num_teacher_nodes: int, num_classes: int, k: int) -> int:
    id_bytes = 2 if int(num_classes) <= 65_535 else 4
    return int(num_teacher_nodes) * (int(k) * id_bytes + int(k) * 2 + 2 + 2 + 2)


def compute_teacher_cache_policy_v2(
    *,
    num_nodes: int,
    num_teacher_nodes: int,
    num_classes: int,
    is_ultra_dataset: bool,
    dense_budget_bytes: int = DEFAULT_TEACHER_CACHE_DENSE_BUDGET_BYTES,
) -> TeacherCacheDecision:
    dense_bytes = int(num_teacher_nodes) * int(num_classes) * 2
    if bool(is_ultra_dataset):
        k = 16 if int(num_classes) > 512 else 8
        return TeacherCacheDecision(
            policy="auto_by_bytes",
            cache_mode=f"topk{k}_tail",
            cache_k=k,
            dense_bytes=dense_bytes,
            cache_bytes=_topk_cache_bytes(num_teacher_nodes, num_classes, k),
            uses_dense_all_node_teacher_cache=False,
        )
    if dense_bytes <= int(dense_budget_bytes):
        return TeacherCacheDecision(
            policy="auto_by_bytes",
            cache_mode="dense_fp16",
            cache_k=0,
            dense_bytes=dense_bytes,
            cache_bytes=dense_bytes,
            uses_dense_all_node_teacher_cache=False,
        )
    k = 8 if int(num_classes) <= 64 else 16
    return TeacherCacheDecision(
        policy="auto_by_bytes",
        cache_mode=f"topk{k}_tail",
        cache_k=k,
        dense_bytes=dense_bytes,
        cache_bytes=_topk_cache_bytes(num_teacher_nodes, num_classes, k),
        uses_dense_all_node_teacher_cache=False,
    )


def selection_weight_schedule_v2(
    *,
    tau: float,
    q_t: float,
    domain_gap_train_all: float,
    domain_gap_threshold: float = 0.10,
    domain_gap_tau: float = 0.05,
) -> dict[str, float]:
    raw = {
        "coverage": 0.45 + 0.35 * (1.0 - float(tau)),
        "hard": 0.20 + 0.35 * (1.0 - float(tau)),
        "domain": 0.05 + 0.35 * _sigmoid((float(domain_gap_train_all) - float(domain_gap_threshold)) / max(float(domain_gap_tau), EPS)),
        "soft": 0.05 + 0.45 * float(tau) * float(q_t),
        "boundary": 0.05 + 0.20 * float(tau) * float(q_t),
        "rare": 0.10,
        "diversity": 0.05,
    }
    if float(q_t) <= 0.0:
        raw["soft"] = 0.0
        raw["boundary"] = 0.0
    return _normalize_selection(raw)


def loss_weight_schedule_v2(*, tau: float, q_t: float, domain_gap_train_all: float) -> tuple[dict[str, float], float]:
    weights = {
        "lambda_hard": 0.25 + 0.75 * (1.0 - float(tau)),
        "lambda_soft": 0.10 + 0.90 * float(tau) * float(q_t),
        "lambda_prior": 0.02 + 0.03 * float(tau),
        "lambda_domain": min(0.20, 0.05 + max(0.0, float(domain_gap_train_all))),
        "lambda_mixup": 0.05 * float(tau),
    }
    if float(q_t) <= 0.0:
        weights["lambda_soft"] = 0.0
    return weights, 1.5 + 2.5 * float(tau)


def student_capacity_schedule_v2(condensed_nodes: int) -> tuple[str, int, int]:
    n = int(condensed_nodes)
    if n < 1_000:
        return "gamlp_like", 128, 220
    if n < 10_000:
        return "gamlp_like", 256, 260
    if n < 100_000:
        return "gamlp_like", 256, 300
    if n < 500_000:
        return "sagn_like", 384, 300
    return "sagn_like", 512, 300


def compute_t40_schedule(
    *,
    condensed_nodes: int,
    num_classes: int,
    teacher_valid_acc: float | None,
    majority_valid_acc: float | None,
    domain_gap_train_all: float,
    num_nodes: int,
    num_teacher_nodes: int | None = None,
    is_ultra_dataset: bool = False,
    dense_cache_budget_bytes: int = DEFAULT_TEACHER_CACHE_DENSE_BUDGET_BYTES,
) -> T40Schedule:
    b, tau = capacity_phase(int(condensed_nodes), int(num_classes))
    q_t = teacher_reliability(teacher_valid_acc, majority_valid_acc)
    selection = selection_weight_schedule_v2(tau=tau, q_t=q_t, domain_gap_train_all=float(domain_gap_train_all))
    loss, temperature = loss_weight_schedule_v2(tau=tau, q_t=q_t, domain_gap_train_all=float(domain_gap_train_all))
    style, hidden, epochs = student_capacity_schedule_v2(int(condensed_nodes))
    teacher_cache = compute_teacher_cache_policy_v2(
        num_nodes=int(num_nodes),
        num_teacher_nodes=int(num_teacher_nodes if num_teacher_nodes is not None else num_nodes),
        num_classes=int(num_classes),
        is_ultra_dataset=bool(is_ultra_dataset),
        dense_budget_bytes=int(dense_cache_budget_bytes),
    )
    return T40Schedule(
        policy_name="auto_base",
        class_capacity_b=b,
        budget_phase=tau,
        teacher_reliability_q=q_t,
        domain_gap_train_all=float(domain_gap_train_all),
        selection_weights=selection,
        loss_weights=loss,
        soft_temperature=temperature,
        student_family="STT-GatedMixer",
        student_internal_style=style,
        hidden_dim=hidden,
        epochs=epochs,
        teacher_cache=teacher_cache,
    )


def _scaled_weights(base: dict[str, float], factors: dict[str, float]) -> dict[str, float]:
    raw = {key: float(value) * float(factors.get(key, 1.0)) for key, value in base.items()}
    return _normalize_selection(raw)


def apply_candidate_policy(schedule: T40Schedule, policy: str) -> T40Schedule:
    policy = str(policy)
    selection = dict(schedule.selection_weights)
    loss = dict(schedule.loss_weights)
    hidden = int(schedule.hidden_dim)
    epochs = int(schedule.epochs)
    style = str(schedule.student_internal_style)
    if policy == "auto_base":
        pass
    elif policy == "coverage_heavy":
        selection = _normalize_selection(
            {
                "coverage": selection.get("coverage", 0.0) * 1.35,
                "hard": selection.get("hard", 0.0) * 1.25,
                "domain": 0.0,
                "soft": 0.0,
                "boundary": 0.0,
                "rare": selection.get("rare", 0.0),
                "diversity": selection.get("diversity", 0.0),
            }
        )
        loss["lambda_hard"] = min(1.25, float(loss["lambda_hard"]) * 1.10)
        loss["lambda_soft"] = float(loss["lambda_soft"]) * 0.75
    elif policy == "domain_coverage":
        selection = _scaled_weights(selection, {"domain": 1.85, "coverage": 1.10, "soft": 0.80, "boundary": 0.80})
        loss["lambda_domain"] = min(0.25, float(loss["lambda_domain"]) * 1.50)
    elif policy == "teacher_transport":
        if float(schedule.teacher_reliability_q) > 0.0:
            selection = _scaled_weights(selection, {"soft": 1.55, "boundary": 1.40, "coverage": 0.90, "hard": 0.85})
            loss["lambda_soft"] = float(loss["lambda_soft"]) * 1.35
            loss["lambda_hard"] = float(loss["lambda_hard"]) * 0.90
    elif policy == "high_fidelity":
        selection = _scaled_weights(selection, {"soft": 1.20, "boundary": 1.20, "domain": 1.10})
        if hidden < 256:
            hidden = 256
        elif hidden < 384 and schedule.budget_phase >= 0.5:
            hidden = 384
        elif hidden < 512 and schedule.budget_phase >= 0.9:
            hidden = 512
        epochs = min(340, max(epochs, epochs + (40 if schedule.budget_phase >= 0.5 else 20)))
        if schedule.budget_phase >= 0.75 and style == "gamlp_like":
            style = "sagn_like"
        loss["lambda_soft"] = float(loss["lambda_soft"]) * 1.15
        loss["lambda_mixup"] = float(loss["lambda_mixup"]) * 1.25
    else:
        raise ValueError(f"unknown T40 candidate policy: {policy}")
    return replace(
        schedule,
        policy_name=policy,
        selection_weights=selection,
        loss_weights=loss,
        hidden_dim=int(hidden),
        epochs=int(epochs),
        student_internal_style=style,
    )


def policy_selection_score(row: dict[str, Any]) -> float:
    valid_acc = float(row.get("valid_acc", 0.0) or 0.0)
    macro = row.get("valid_macro_f1", "")
    macro_term = 0.0 if macro in {"", None} else 0.10 * float(macro)
    prior = float(row.get("selected_prior_kl", 0.0) or 0.0)
    domain_gap = float(row.get("domain_coverage_gap", 0.0) or 0.0)
    return float(valid_acc + macro_term - 0.05 * prior - 0.05 * domain_gap)


def select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate rows cannot be empty")
    scored: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["policy_selection_score"] = policy_selection_score(out)
        scored.append(out)
    return max(scored, key=lambda item: (float(item["policy_selection_score"]), float(item.get("valid_acc", 0.0) or 0.0), str(item.get("selected_policy", ""))))


def schedule_to_row_fields_v2(schedule: T40Schedule) -> dict[str, Any]:
    return {
        "budget_phase": schedule.budget_phase,
        "class_capacity_b": schedule.class_capacity_b,
        "teacher_reliability_q": schedule.teacher_reliability_q,
        "teacher_cache_policy": schedule.teacher_cache.policy,
        "teacher_cache_mode": schedule.teacher_cache.cache_mode,
        "teacher_cache_k": schedule.teacher_cache.cache_k,
        "teacher_cache_bytes": schedule.teacher_cache.cache_bytes,
        "uses_dense_teacher_cache": schedule.teacher_cache.cache_mode == "dense_fp16",
        "uses_dense_all_node_teacher_cache": schedule.teacher_cache.uses_dense_all_node_teacher_cache,
        "coverage_weight": schedule.selection_weights["coverage"],
        "hard_anchor_weight": schedule.selection_weights["hard"],
        "domain_weight": schedule.selection_weights["domain"],
        "soft_teacher_weight": schedule.selection_weights["soft"],
        "boundary_weight": schedule.selection_weights["boundary"],
        "rare_weight": schedule.selection_weights["rare"],
        "mixup_weight": schedule.loss_weights["lambda_mixup"],
        "student_family": schedule.student_family,
        "student_internal_style": schedule.student_internal_style,
        "student_capacity": f"{schedule.student_family}:{schedule.student_internal_style}:h{schedule.hidden_dim}:e{schedule.epochs}",
        "hidden_dim": schedule.hidden_dim,
        "epochs": schedule.epochs,
        "soft_temperature": schedule.soft_temperature,
        "lambda_hard": schedule.loss_weights["lambda_hard"],
        "lambda_soft": schedule.loss_weights["lambda_soft"],
        "lambda_prior": schedule.loss_weights["lambda_prior"],
        "lambda_domain": schedule.loss_weights["lambda_domain"],
        "lambda_mixup": schedule.loss_weights["lambda_mixup"],
        "domain_gap_train_all": schedule.domain_gap_train_all,
    }
