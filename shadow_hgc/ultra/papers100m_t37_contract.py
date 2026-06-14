from __future__ import annotations

from collections import Counter
from typing import Any

from shadow_hgc.ultra.papers100m_contract import PAPERS100M_NUM_NODES, truthy
from shadow_hgc.ultra.papers100m_t36_contract import DISCO_PARITY_RATIOS, SCALE_FIDELITY_RATIOS, ratio_matches


T37_DISCO_PARITY_RATIOS: tuple[float, ...] = DISCO_PARITY_RATIOS
T37_NATIVE_RATIOS: tuple[float, ...] = SCALE_FIDELITY_RATIOS


T36_RANDOM_ONECACHE_SGC: dict[float, float] = {
    0.00005: 0.44862786813350874,
    0.00010: 0.4991228806837798,
    0.00020: 0.5313850087245379,
    0.00050: 0.5665584264106225,
}


T36_NATIVE_REFERENCE: dict[float, float] = {
    0.00050: 0.4738683761162276,
    0.00100: 0.5273446612359918,
    0.00200: 0.5674915320661759,
    0.00500: 0.604335208875701,
    0.01000: 0.6197407832488873,
}


T37_REQUIRED_FIELDS: list[str] = [
    "stage",
    "dataset",
    "method",
    "seed",
    "backend",
    "comparison_type",
    "requested_full_node_ratio",
    "ratio_percent",
    "full_node_denominator",
    "condensed_nodes",
    "target_universe_size",
    "target_universe_ratio",
    "valid_acc",
    "accuracy",
    "macro_f1",
    "predicted_classes",
    "disco_acc",
    "random_onecache_acc",
    "beats_disco",
    "beats_random_onecache",
    "relative_error_reduction_vs_disco",
    "relative_error_reduction_vs_random",
    "cache_build_id",
    "edge_cache_id",
    "sft_cache_id",
    "teacher_cache_id",
    "selection_bank_id",
    "edge_slice_cache_reused",
    "sft_cache_reused",
    "teacher_cache_reused",
    "selection_bank_reused",
    "incremental_edge_scans_after_cache_build",
    "bank_policy",
    "bank_max_ratio",
    "candidate_universe",
    "coverage_axes",
    "year_bucket_available",
    "degree_bucket_mode",
    "feature_bucket_mode",
    "feature_lsh_dim",
    "feature_lsh_bits",
    "teacher_weight_eta",
    "class_floor_requested",
    "class_floor_actual_min",
    "class_floor_violation_count",
    "prefix_overlap_with_previous_ratio",
    "prefix_violation_count",
    "selected_count",
    "selected_class_count",
    "selected_predicted_class_count",
    "selected_train_anchor_count",
    "selected_soft_prior_kl",
    "selected_hard_label_prior_kl",
    "coverage_bucket_count",
    "empty_bucket_count",
    "student_model",
    "hidden_dim",
    "epochs",
    "temperature",
    "lambda_hard",
    "lambda_soft",
    "lambda_prior",
    "uses_teacher_probs_as_soft_targets",
    "uses_teacher_weighting",
    "uses_dense_teacher_cache_in_ram",
    "uses_dense_all_node_teacher_cache",
    "uses_full_edge_index_on_gpu",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_exact_all_pair",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "materialize_time",
    "student_train_time",
    "eval_time",
    "condensed_bytes",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "promotion_status",
    "failure_reason",
    "notes",
]


T37_FORBIDDEN_PROMOTED_FLAGS: tuple[str, ...] = (
    "uses_dense_teacher_cache_in_ram",
    "uses_dense_all_node_teacher_cache",
    "uses_full_edge_index_on_gpu",
    "uses_e_by_d_materialization",
    "uses_dense_p2",
    "uses_exact_all_pair",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
)


def _ratio_percent(ratio: float) -> float:
    return float(ratio) * 100.0


def comparison_type_for_t37(backend: str, ratio: float) -> str:
    if str(backend).lower() == "sgc" and ratio_matches(float(ratio), T37_DISCO_PARITY_RATIOS):
        return "disco_parity"
    if str(backend).lower() in {"gamlp_table", "sagn_table"} and ratio_matches(float(ratio), T37_NATIVE_RATIOS):
        return "ours_native_scale_fidelity"
    return "diagnostic"


def make_t37_row(
    *,
    stage: str = "T37",
    dataset: str = "ogbn-papers100M",
    method: str = "",
    seed: int = 42,
    backend: str = "",
    comparison_type: str | None = None,
    requested_full_node_ratio: float = 0.0,
    full_node_denominator: int = PAPERS100M_NUM_NODES,
    condensed_nodes: int = 0,
    target_universe_size: int = 0,
    promotion_status: str = "diagnostic",
    failure_reason: str = "",
    **fields: Any,
) -> dict[str, Any]:
    ratio = float(requested_full_node_ratio)
    target_size = int(target_universe_size)
    row: dict[str, Any] = {
        "stage": stage,
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "backend": str(backend).lower(),
        "comparison_type": comparison_type or comparison_type_for_t37(backend, ratio),
        "requested_full_node_ratio": ratio,
        "ratio_percent": _ratio_percent(ratio),
        "full_node_denominator": int(full_node_denominator),
        "condensed_nodes": int(condensed_nodes),
        "target_universe_size": target_size,
        "target_universe_ratio": float(condensed_nodes) / float(target_size) if target_size else 0.0,
        "valid_acc": "",
        "accuracy": "",
        "macro_f1": "",
        "predicted_classes": "",
        "disco_acc": "",
        "random_onecache_acc": "",
        "beats_disco": "",
        "beats_random_onecache": "",
        "relative_error_reduction_vs_disco": "",
        "relative_error_reduction_vs_random": "",
        "cache_build_id": "",
        "edge_cache_id": "",
        "sft_cache_id": "",
        "teacher_cache_id": "",
        "selection_bank_id": "",
        "edge_slice_cache_reused": True,
        "sft_cache_reused": True,
        "teacher_cache_reused": True,
        "selection_bank_reused": True,
        "incremental_edge_scans_after_cache_build": 0,
        "bank_policy": "",
        "bank_max_ratio": "",
        "candidate_universe": "train_targets",
        "coverage_axes": "",
        "year_bucket_available": False,
        "degree_bucket_mode": "log2",
        "feature_bucket_mode": "",
        "feature_lsh_dim": "",
        "feature_lsh_bits": "",
        "teacher_weight_eta": 0.0,
        "class_floor_requested": "",
        "class_floor_actual_min": "",
        "class_floor_violation_count": "",
        "prefix_overlap_with_previous_ratio": "",
        "prefix_violation_count": 0,
        "selected_count": "",
        "selected_class_count": "",
        "selected_predicted_class_count": "",
        "selected_train_anchor_count": "",
        "selected_soft_prior_kl": "",
        "selected_hard_label_prior_kl": "",
        "coverage_bucket_count": "",
        "empty_bucket_count": "",
        "student_model": "",
        "hidden_dim": "",
        "epochs": "",
        "temperature": "",
        "lambda_hard": "",
        "lambda_soft": "",
        "lambda_prior": "",
        "uses_teacher_probs_as_soft_targets": False,
        "uses_teacher_weighting": False,
        "uses_dense_teacher_cache_in_ram": False,
        "uses_dense_all_node_teacher_cache": False,
        "uses_full_edge_index_on_gpu": False,
        "uses_e_by_d_materialization": False,
        "uses_dense_p2": False,
        "uses_exact_all_pair": False,
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
        "materialize_time": "",
        "student_train_time": "",
        "eval_time": "",
        "condensed_bytes": "",
        "peak_cpu_ram": "",
        "peak_gpu_ram": "",
        "promotion_status": promotion_status,
        "failure_reason": failure_reason,
        "notes": "",
    }
    row.update(fields)
    for field in T37_REQUIRED_FIELDS:
        row.setdefault(field, "")
    return row


def validate_t37_row(row: dict[str, Any]) -> dict[str, Any]:
    flags = [f"missing_field:{field}" for field in T37_REQUIRED_FIELDS if field not in row]
    promoted = str(row.get("promotion_status", "")).lower() == "promoted"
    if promoted:
        for flag in T37_FORBIDDEN_PROMOTED_FLAGS:
            if truthy(row.get(flag, False)):
                flags.append(flag)
        for flag in ("edge_slice_cache_reused", "sft_cache_reused", "teacher_cache_reused"):
            if not truthy(row.get(flag, False)):
                flags.append(f"{flag}_false")
        if not truthy(row.get("selection_bank_reused", False)) and str(row.get("comparison_type")) != "bank_build":
            flags.append("selection_bank_reused_false")
        if int(row.get("incremental_edge_scans_after_cache_build", 0) or 0) != 0:
            flags.append("incremental_edge_scans_after_cache_build_nonzero")
        ratio = float(row.get("requested_full_node_ratio", 0.0) or 0.0)
        backend = str(row.get("backend", "")).lower()
        comparison = str(row.get("comparison_type", ""))
        if comparison == "disco_parity":
            if backend != "sgc":
                flags.append("disco_parity_backend_not_sgc")
            if not ratio_matches(ratio, T37_DISCO_PARITY_RATIOS):
                flags.append("disco_parity_ratio_mismatch")
            if int(row.get("full_node_denominator", 0) or 0) != PAPERS100M_NUM_NODES:
                flags.append("disco_parity_denominator_mismatch")
        if backend in {"gamlp_table", "sagn_table"} and comparison == "disco_parity":
            flags.append("native_backend_marked_disco_parity")
    return {"valid": not flags, "forbidden_flags": sorted(set(flags))}


def attach_t37_reference_metrics(row: dict[str, Any], refs: dict[float, dict[str, Any]]) -> dict[str, Any]:
    out = dict(row)
    ratio = float(out.get("requested_full_node_ratio", 0.0) or 0.0)
    ref = refs.get(ratio)
    if not ref:
        return out
    acc_value = out.get("accuracy", "")
    out["disco_acc"] = float(ref.get("disco_acc", "nan"))
    out["random_onecache_acc"] = float(ref.get("random_onecache_acc", ref.get("random_acc", "nan")))
    if acc_value != "":
        acc = float(acc_value)
        disco = float(out["disco_acc"])
        random_acc = float(out["random_onecache_acc"])
        out["beats_disco"] = acc >= disco
        out["beats_random_onecache"] = acc >= random_acc
        out["relative_error_reduction_vs_disco"] = (acc - disco) / max(1.0 - disco, 1e-12)
        out["relative_error_reduction_vs_random"] = (acc - random_acc) / max(1.0 - random_acc, 1e-12)
    return out


def audit_one_cache_reuse_t37(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if not rows:
        reasons.append("rows_missing")
    for key in ("edge_cache_id", "sft_cache_id", "teacher_cache_id"):
        values = {str(row.get(key, "")) for row in rows if str(row.get(key, ""))}
        if len(values) > 1:
            reasons.append(f"{key}_mismatch")
        if rows and not values:
            reasons.append(f"{key}_missing")
    for row in rows:
        if int(row.get("incremental_edge_scans_after_cache_build", 0) or 0) != 0:
            reasons.append("incremental_edge_scans_after_cache_build_nonzero")
        for flag in ("edge_slice_cache_reused", "sft_cache_reused", "teacher_cache_reused", "selection_bank_reused"):
            if not truthy(row.get(flag, False)):
                reasons.append(f"{flag}_false")
    return {"valid": not reasons, "failure_reasons": sorted(set(reasons)), "reason_counts": dict(Counter(reasons))}


def summarize_t37_stage(
    *,
    disco_rows: list[dict[str, Any]],
    native_rows: list[dict[str, Any]],
    multiseed_rows: list[dict[str, Any]],
    bank_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    checked = disco_rows + native_rows + multiseed_rows
    forbidden = [validate_t37_row(row) for row in checked]
    promoted_disco = [row for row in disco_rows if str(row.get("promotion_status", "")).lower() == "promoted"]
    beats_disco = sum(1 for row in promoted_disco if truthy(row.get("beats_disco", False)))
    beats_random = sum(1 for row in promoted_disco if truthy(row.get("beats_random_onecache", False)))
    low_ratio_ok = any(
        abs(float(row.get("requested_full_node_ratio", 0.0) or 0.0) - 0.00005) < 1e-12
        and float(row.get("accuracy", 0.0) or 0.0) >= 0.483
        for row in promoted_disco
    )
    native_05_ok = any(
        abs(float(row.get("requested_full_node_ratio", 0.0) or 0.0) - 0.005) < 1e-12
        and float(row.get("accuracy", 0.0) or 0.0) > T36_NATIVE_REFERENCE[0.005]
        for row in native_rows
    )
    native_10_ok = any(
        abs(float(row.get("requested_full_node_ratio", 0.0) or 0.0) - 0.01) < 1e-12
        and float(row.get("accuracy", 0.0) or 0.0) > T36_NATIVE_REFERENCE[0.01]
        for row in native_rows
    )
    no_rebuild = audit_one_cache_reuse_t37(checked)
    disco_success = beats_disco >= 4 and beats_random >= 3 and low_ratio_ok
    stop = (not disco_success) or (not low_ratio_ok) or (not native_05_ok and not native_10_ok) or (not no_rebuild["valid"])
    return {
        "teacher_rows_present": bool(teacher_rows),
        "bank_rows_present": bool(bank_rows),
        "disco_rows_present": bool(disco_rows),
        "native_rows_present": bool(native_rows),
        "multiseed_rows_present": bool(multiseed_rows),
        "disco_beats_disco_count": beats_disco,
        "disco_beats_random_count": beats_random,
        "disco_0p005_reaches_disco": low_ratio_ok,
        "disco_success_gate": disco_success,
        "native_0p5_improves_t36": native_05_ok,
        "native_1p0_improves_t36": native_10_ok,
        "no_rebuild_gate": bool(no_rebuild["valid"]),
        "forbidden_guard_hits": sum(0 if item["valid"] else 1 for item in forbidden),
        "stop_condition_met": stop,
        "row_count_disco": len(disco_rows),
        "row_count_native": len(native_rows),
        "row_count_multiseed": len(multiseed_rows),
        "row_count_bank": len(bank_rows),
        "row_count_teacher": len(teacher_rows),
        "no_rebuild_failure_reasons": ",".join(no_rebuild["failure_reasons"]),
    }
