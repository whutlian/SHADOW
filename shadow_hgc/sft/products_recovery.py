from __future__ import annotations

from typing import Any

from shadow_hgc.ratio.scale_bucket import account_full_node_ratio, validate_t24_promoted_row


PRODUCTS_FULLGRAPH_TEACHER = {
    "variant": "P7_sagn_lite_v2",
    "accuracy": 0.7555780580193042,
    "macro_f1": 0.4046991170720907,
}


def validate_products_recovery_row(row: dict[str, Any]) -> dict[str, Any]:
    result = validate_t24_promoted_row(row)
    if str(row.get("promotion_status", "")).startswith("promoted") and str(row.get("status", "")).endswith("proxy"):
        flags = list(result["forbidden_flags"])
        if "proxy_promotion" not in flags:
            flags.append("proxy_promotion")
        return {"valid": False, "forbidden_flags": flags}
    return result


def products_recovery_row(
    *,
    ratio: float,
    method: str,
    status: str,
    accuracy: float | str,
    macro_f1: float | str,
    target_prototypes: int,
    shadow_nodes: int,
    condensed_edges: int,
    original_total_nodes: int = 2_449_029,
    feature_cache_bytes: int = 0,
    is_proxy: bool = False,
    promotion_status: str = "not_promoted",
    reason: str = "",
) -> dict[str, Any]:
    accounting = account_full_node_ratio(
        original_total_nodes=int(original_total_nodes),
        target_prototypes=int(target_prototypes),
        shadow_nodes=int(shadow_nodes),
        condensed_edges=int(condensed_edges),
    )
    full_acc = float(PRODUCTS_FULLGRAPH_TEACHER["accuracy"])
    acc_value = "" if accuracy == "" else float(accuracy)
    row = {
        "dataset": "ogbn-products",
        "method": method,
        "requested_full_node_ratio": float(ratio),
        "scale_bucket": "large",
        "status": status,
        "fullgraph_acc": full_acc,
        "identity_acc": full_acc if method == "P0_identity_replay" else "",
        "prototype_oracle_acc": acc_value if "prototype_oracle" in method else "",
        "shadow_condensed_acc": acc_value if "shadow" in method else "",
        "accuracy": acc_value,
        "macro_f1": macro_f1,
        "full_to_identity_gap": 0.0 if method == "P0_identity_replay" else "",
        "identity_to_oracle_gap": "" if "prototype_oracle" not in method or acc_value == "" else full_acc - float(acc_value),
        "oracle_to_shadow_gap": "",
        "full_to_shadow_gap": "" if "shadow" not in method or acc_value == "" else full_acc - float(acc_value),
        "feature_cache_bytes": int(feature_cache_bytes),
        "condensation_time_s": "",
        "training_time_s": "",
        "inference_time_s": "",
        "peak_cpu_ram_gb": "",
        "peak_gpu_ram_gb": "",
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_legacy_diffusion": False,
        "uses_coverage_medoid": False,
        "uses_source_anchors": False,
        "uses_bounded_edges": False,
        "uses_e_by_d": False,
        "is_proxy": bool(is_proxy),
        "promotion_status": promotion_status,
        "promotion_reason": reason,
        **accounting,
    }
    row["requested_full_node_ratio"] = float(ratio)
    safety = validate_products_recovery_row(row)
    if not safety["valid"]:
        row["promotion_status"] = "blocked_forbidden"
        row["promotion_reason"] = ",".join(safety["forbidden_flags"])
    return row
