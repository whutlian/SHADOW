from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


MI_B = 1024 * 1024
DEFAULT_TEACHER_CACHE_DENSE_BUDGET_BYTES = 256 * MI_B


@dataclass(frozen=True)
class TeacherCacheDecision:
    policy: str
    cache_mode: str
    cache_k: int
    dense_bytes: int
    cache_bytes: int
    uses_dense_all_node_teacher_cache: bool


@dataclass(frozen=True)
class UnifiedSchedule:
    budget_per_class: float
    budget_phase_tau: float
    teacher_reliability_q: float
    selection_weights: dict[str, float]
    loss_weights: dict[str, float]
    soft_temperature: float
    student_family: str
    student_internal_style: str
    hidden_dim: int
    epochs: int
    teacher_cache: TeacherCacheDecision


def clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def budget_phase_tau(condensed_nodes: int, num_classes: int) -> tuple[float, float]:
    b = max(1.0, float(condensed_nodes) / float(max(1, num_classes)))
    tau = clip((math.log2(b) - math.log2(16.0)) / (math.log2(256.0) - math.log2(16.0)), 0.0, 1.0)
    return b, tau


def teacher_reliability_q(
    teacher_valid_acc: float | None,
    majority_valid_acc: float | None,
    *,
    eps: float = 1e-12,
) -> float:
    if teacher_valid_acc is None or majority_valid_acc is None:
        return 0.0
    return clip((float(teacher_valid_acc) - float(majority_valid_acc)) / (1.0 - float(majority_valid_acc) + eps), 0.0, 1.0)


def selection_weight_schedule(tau: float, q_t: float) -> dict[str, float]:
    raw = {
        "coverage": 0.45 + 0.35 * (1.0 - tau),
        "hard": 0.20 + 0.35 * (1.0 - tau),
        "soft": 0.05 + 0.45 * tau * q_t,
        "boundary": 0.05 + 0.20 * tau * q_t,
        "rare": 0.10,
        "diversity": 0.05,
    }
    if q_t <= 0.0:
        raw["soft"] = 0.0
        raw["boundary"] = 0.0
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def loss_weight_schedule(tau: float, q_t: float) -> tuple[dict[str, float], float]:
    weights = {
        "alpha_hard": 0.25 + 0.75 * (1.0 - tau),
        "alpha_soft": 0.10 + 0.90 * tau * q_t,
        "alpha_prior": 0.02 + 0.03 * tau,
        "alpha_mix": 0.05 * tau,
    }
    if q_t <= 0.0:
        weights["alpha_soft"] = 0.0
    return weights, 1.5 + 2.5 * tau


def student_capacity_schedule(condensed_nodes: int, tau: float) -> tuple[int, int, str]:
    if condensed_nodes < 1_000:
        hidden_dim = 128
    elif condensed_nodes < 50_000:
        hidden_dim = 256
    elif condensed_nodes < 500_000:
        hidden_dim = 384
    else:
        hidden_dim = 512

    if condensed_nodes < 10_000:
        epochs = 220
    elif condensed_nodes < 100_000:
        epochs = 260
    else:
        epochs = 300

    if tau < 0.35:
        style = "gamlp_like"
    elif condensed_nodes >= 50_000:
        style = "sagn_like"
    else:
        style = "gamlp_like"
    return hidden_dim, epochs, style


def _topk_cache_bytes(num_teacher_nodes: int, num_classes: int, k: int) -> int:
    id_bytes = 2 if int(num_classes) <= 65_535 else 4
    # top-k ids, top-k fp16 probabilities, fp16 tail mass, entropy, and margin.
    return int(num_teacher_nodes) * (int(k) * id_bytes + int(k) * 2 + 2 + 2 + 2)


def compute_teacher_cache_policy(
    *,
    num_nodes: int,
    num_teacher_nodes: int,
    num_classes: int,
    dense_budget_bytes: int = DEFAULT_TEACHER_CACHE_DENSE_BUDGET_BYTES,
) -> TeacherCacheDecision:
    del dense_budget_bytes
    dense_bytes = int(num_teacher_nodes) * int(num_classes) * 2
    k = 4 if int(num_classes) <= 64 else 8
    mode = f"topk{k}_tail"
    cache_bytes = _topk_cache_bytes(num_teacher_nodes, num_classes, k)
    return TeacherCacheDecision(
        policy="auto_topk_by_class",
        cache_mode=mode,
        cache_k=k,
        dense_bytes=dense_bytes,
        cache_bytes=cache_bytes,
        uses_dense_all_node_teacher_cache=False,
    )


def compute_unified_schedule(
    *,
    condensed_nodes: int,
    num_classes: int,
    teacher_valid_acc: float | None,
    majority_valid_acc: float | None,
    num_nodes: int,
    num_teacher_nodes: int | None = None,
    teacher_cache_dense_budget_bytes: int = DEFAULT_TEACHER_CACHE_DENSE_BUDGET_BYTES,
) -> UnifiedSchedule:
    b, tau = budget_phase_tau(int(condensed_nodes), int(num_classes))
    q_t = teacher_reliability_q(teacher_valid_acc, majority_valid_acc)
    selection = selection_weight_schedule(tau, q_t)
    loss, temperature = loss_weight_schedule(tau, q_t)
    hidden_dim, epochs, style = student_capacity_schedule(int(condensed_nodes), tau)
    teacher_cache = compute_teacher_cache_policy(
        num_nodes=int(num_nodes),
        num_teacher_nodes=int(num_teacher_nodes if num_teacher_nodes is not None else num_nodes),
        num_classes=int(num_classes),
        dense_budget_bytes=int(teacher_cache_dense_budget_bytes),
    )
    return UnifiedSchedule(
        budget_per_class=b,
        budget_phase_tau=tau,
        teacher_reliability_q=q_t,
        selection_weights=selection,
        loss_weights=loss,
        soft_temperature=temperature,
        student_family="stt_gated_mixer",
        student_internal_style=style,
        hidden_dim=hidden_dim,
        epochs=epochs,
        teacher_cache=teacher_cache,
    )


def full_node_ratio(*, condensed_nodes: int, original_num_nodes: int) -> float:
    if int(original_num_nodes) <= 0:
        return 0.0
    return float(condensed_nodes) / float(original_num_nodes)


def schedule_to_row_fields(schedule: UnifiedSchedule) -> dict[str, Any]:
    return {
        "budget_per_class": schedule.budget_per_class,
        "budget_phase_tau": schedule.budget_phase_tau,
        "budget_phase": schedule.budget_phase_tau,
        "teacher_reliability_q": schedule.teacher_reliability_q,
        "teacher_cache_policy": schedule.teacher_cache.policy,
        "teacher_cache_mode": schedule.teacher_cache.cache_mode,
        "teacher_cache_k": schedule.teacher_cache.cache_k,
        "teacher_cache_bytes": schedule.teacher_cache.cache_bytes,
        "coverage_weight": schedule.selection_weights["coverage"],
        "hard_weight": schedule.selection_weights["hard"],
        "soft_weight": schedule.selection_weights["soft"],
        "boundary_weight": schedule.selection_weights["boundary"],
        "rare_weight": schedule.selection_weights["rare"],
        "diversity_weight": schedule.selection_weights["diversity"],
        "alpha_hard": schedule.loss_weights["alpha_hard"],
        "alpha_soft": schedule.loss_weights["alpha_soft"],
        "alpha_prior": schedule.loss_weights["alpha_prior"],
        "alpha_mix": schedule.loss_weights["alpha_mix"],
        "soft_temperature": schedule.soft_temperature,
        "student_family": schedule.student_family,
        "student_internal_style": schedule.student_internal_style,
        "student_capacity": f"{schedule.student_family}:{schedule.student_internal_style}:h{schedule.hidden_dim}:e{schedule.epochs}",
        "hidden_dim": schedule.hidden_dim,
        "epochs": schedule.epochs,
        "uses_dense_all_node_teacher_cache": schedule.teacher_cache.uses_dense_all_node_teacher_cache,
    }
