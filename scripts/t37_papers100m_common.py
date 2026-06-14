from __future__ import annotations

from pathlib import Path
from typing import Any

from shadow_hgc.ultra.papers100m_condensed import materialize_condensed_table, train_and_eval_condensed_table
from shadow_hgc.ultra.papers100m_disco_parity import ensure_disco_baseline_csv, load_disco_baseline
from shadow_hgc.ultra.papers100m_memmap import directory_bytes
from shadow_hgc.ultra.papers100m_nested_bank import NESTED_BANK_POLICY, build_external_onecache_bank, build_nested_bank_v2
from shadow_hgc.ultra.papers100m_ratio_policy_v2 import ratio_policy_v2
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_scr_bank import (
    SCR_POLICIES,
    SCR_POLICY_FULL,
    SCR_POLICY_FULL_TEACHER_WEIGHT,
    audit_scr_bank,
    build_scr_bank,
    policy_to_bank_policy,
)
from shadow_hgc.ultra.papers100m_sgc_backend import train_and_eval_sgc_condensed
from shadow_hgc.ultra.papers100m_t37_contract import T36_RANDOM_ONECACHE_SGC, attach_t37_reference_metrics, make_t37_row, validate_t37_row


def load_t37_disco_references(path: str | Path) -> dict[float, dict[str, Any]]:
    ensure_disco_baseline_csv(path)
    disco = load_disco_baseline(path)
    refs: dict[float, dict[str, Any]] = {}
    for ratio, row in disco.items():
        refs[float(ratio)] = {
            "disco_acc": float(row["disco_acc"]),
            "random_onecache_acc": float(T36_RANDOM_ONECACHE_SGC.get(float(ratio), row["random_acc"])),
            "public_random_acc": float(row["random_acc"]),
            "herding_acc": float(row["herding_acc"]),
            "kcenter_acc": float(row["kcenter_acc"]),
        }
    return refs


def t37_policy_for_method(method: str, *, teacher_weight_eta: float = 0.10) -> str:
    text = str(method)
    if text == "random_onecache":
        return "random_onecache_t36"
    if text in {"stt_nested_bank_v2", "stt_current_t35"}:
        return NESTED_BANK_POLICY if text == "stt_nested_bank_v2" else "stt_ratio_v2"
    if text in {"stt_randcore_gamlp", "stt_randcore_dual_loss", "stt_randcore_sagn"}:
        return SCR_POLICY_FULL
    if text == "stt_randcore_teacher_weighted":
        return policy_to_bank_policy(SCR_POLICY_FULL_TEACHER_WEIGHT, teacher_weight_eta=teacher_weight_eta)
    if text in SCR_POLICIES:
        return policy_to_bank_policy(text, teacher_weight_eta=teacher_weight_eta)
    return text


def ensure_t37_bank(
    cache_root: str | Path,
    *,
    method: str,
    seed: int,
    max_ratio: float,
    feature_lsh_dim: int = 64,
    feature_lsh_bits: int = 16,
    teacher_weight_eta: float = 0.10,
    force: bool = False,
) -> tuple[str, dict[str, Any]]:
    cache_root = Path(cache_root)
    policy = t37_policy_for_method(method, teacher_weight_eta=teacher_weight_eta)
    ctx = Papers100MCacheContext(cache_root, selection_policy=policy, seed=int(seed))
    if method == "random_onecache":
        return policy, build_external_onecache_bank(ctx, method="random_onecache", seed=int(seed), max_ratio=float(max_ratio), force=force)
    if method == "stt_nested_bank_v2":
        return policy, build_nested_bank_v2(ctx, policy=policy, seed=int(seed), max_ratio=float(max_ratio), teacher_id="best_t36_teacher", force=force)
    if method == "stt_current_t35":
        return policy, ctx.bank
    if method in SCR_POLICIES or str(method).startswith("stt_randcore"):
        return policy, build_scr_bank(
            ctx,
            policy=policy,
            seed=int(seed),
            max_ratio=float(max_ratio),
            feature_lsh_dim=int(feature_lsh_dim),
            feature_lsh_bits=int(feature_lsh_bits),
            teacher_weight_eta=float(teacher_weight_eta) if (method == SCR_POLICY_FULL_TEACHER_WEIGHT or method == "stt_randcore_teacher_weighted") else 0.0,
            force=force,
        )
    raise ValueError(f"unknown T37 method: {method}")


def t37_materialized_row(
    ctx: Papers100MCacheContext,
    *,
    method: str,
    backend: str,
    ratio: float,
    seed: int,
    policy: str,
    bank_manifest: dict[str, Any],
    materialized: dict[str, Any],
    comparison_type: str,
    audit_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ids = ctx.cache_ids()
    bank = bank_manifest or {}
    audit = audit_row or {}
    row = make_t37_row(
        method=str(method),
        seed=int(seed),
        backend=str(backend).lower(),
        comparison_type=str(comparison_type),
        requested_full_node_ratio=float(ratio),
        full_node_denominator=int(ctx.manifest["num_nodes"]),
        condensed_nodes=int(materialized.get("condensed_nodes", 0)),
        target_universe_size=int(ctx.manifest["target_universe_size"]),
        cache_build_id=ids["cache_build_id"],
        edge_cache_id=ids["edge_slice_cache_id"],
        sft_cache_id=ids["sft_cache_id"],
        teacher_cache_id=ids["teacher_cache_id"],
        selection_bank_id=ids["selection_bank_id"],
        bank_policy=policy,
        bank_max_ratio=bank.get("max_ratio_for_bank", bank.get("max_ratio", "")),
        candidate_universe=bank.get("candidate_universe", "train_targets" if str(method).startswith("scr") or str(method).startswith("stt_randcore") else "all_targets"),
        coverage_axes=bank.get("coverage_axes", ""),
        year_bucket_available=bank.get("year_bucket_available", False),
        degree_bucket_mode=bank.get("degree_bucket_mode", ""),
        feature_bucket_mode=bank.get("feature_bucket_mode", ""),
        feature_lsh_dim=bank.get("feature_lsh_dim", ""),
        feature_lsh_bits=bank.get("feature_lsh_bits", ""),
        teacher_weight_eta=bank.get("teacher_weight_eta", 0.0),
        class_floor_requested=audit.get("class_floor_requested", bank.get("class_floor_requested", "")),
        class_floor_actual_min=audit.get("class_floor_actual_min", bank.get("class_floor_actual_min", "")),
        class_floor_violation_count=audit.get("class_floor_violation_count", bank.get("class_floor_violation_count", "")),
        prefix_overlap_with_previous_ratio=audit.get("prefix_overlap_with_previous_ratio", ""),
        prefix_violation_count=audit.get("prefix_violation_count", 0),
        selected_count=audit.get("selected_count", int(materialized.get("condensed_nodes", 0))),
        selected_class_count=audit.get("selected_class_count", ""),
        selected_predicted_class_count=audit.get("selected_predicted_class_count", ""),
        selected_train_anchor_count=audit.get("selected_train_anchor_count", ""),
        selected_soft_prior_kl=audit.get("selected_soft_prior_kl", ""),
        selected_hard_label_prior_kl=audit.get("selected_hard_label_prior_kl", ""),
        coverage_bucket_count=audit.get("coverage_bucket_count", bank.get("coverage_bucket_count", "")),
        empty_bucket_count=audit.get("empty_bucket_count", bank.get("empty_bucket_count", "")),
        materialize_time=materialized.get("condensed_materialize_time", ""),
        condensed_bytes=materialized.get("condensed_cache_bytes", ""),
        uses_teacher_weighting=bool(bank.get("uses_teacher_weighting", False)),
        notes=f"selection_policy={policy}",
    )
    return row


def audit_rows_for_policy(cache_root: str | Path, *, policy: str, seed: int, ratios: list[float]) -> dict[float, dict[str, Any]]:
    try:
        return {float(row["ratio"]): row for row in audit_scr_bank(cache_root, policy=policy, seed=int(seed), ratios=[float(v) for v in ratios])}
    except FileNotFoundError:
        return {}


def run_t37_backend(
    ctx: Papers100MCacheContext,
    *,
    ratio: float,
    backend: str,
    method: str,
    device: str,
) -> dict[str, Any]:
    backend_name = str(backend).lower()
    policy = ratio_policy_v2(float(ratio))
    if backend_name == "sgc":
        return train_and_eval_sgc_condensed(
            ctx,
            ratio,
            epochs=180,
            temperature=1.0,
            lambda_hard=1.0,
            lambda_soft=0.0,
            lambda_prior=0.02,
            device=device,
            use_soft_targets=False,
        )
    if backend_name == "gamlp_table":
        lambda_hard = 0.5
        lambda_soft = 0.5
        temperature = 2.0
        lambda_prior = 0.01
        if method == "stt_randcore_dual_loss":
            if float(ratio) <= 0.001:
                lambda_hard, lambda_soft, temperature = 0.75, 0.25, 1.5
            elif float(ratio) <= 0.002:
                lambda_hard, lambda_soft, temperature = 0.5, 0.5, 2.0
            else:
                lambda_hard, lambda_soft, temperature = 0.25, 0.75, 2.0
        return train_and_eval_condensed_table(
            ctx,
            ratio,
            student="papers100m_gamlp_table",
            hidden_dim=512,
            epochs=300,
            temperature=temperature,
            lambda_hard=lambda_hard,
            lambda_soft=lambda_soft,
            lambda_prior=lambda_prior,
            device=device,
        )
    if backend_name == "sagn_table":
        return train_and_eval_condensed_table(
            ctx,
            ratio,
            student="papers100m_sagn_table",
            hidden_dim=384,
            epochs=260,
            temperature=4.0,
            lambda_hard=0.5,
            lambda_soft=0.5,
            lambda_prior=0.05,
            device=device,
        )
    raise ValueError(f"unknown backend: {backend}")


def finalize_t37_row(row: dict[str, Any], *, refs: dict[float, dict[str, Any]] | None = None, promoted: bool = True) -> dict[str, Any]:
    out = dict(row)
    if refs:
        out = attach_t37_reference_metrics(out, refs)
    out["condensed_bytes"] = out.get("condensed_bytes", "") or 0
    out["promotion_status"] = "promoted" if promoted else out.get("promotion_status", "diagnostic")
    check = validate_t37_row(out)
    if out["promotion_status"] == "promoted" and not check["valid"]:
        out["promotion_status"] = "not_promoted"
        out["failure_reason"] = ",".join(check["forbidden_flags"])
    return out


def cache_directory_bytes(cache_root: str | Path, ratio: float) -> int:
    name = f"ratio={float(ratio):.12g}".replace("+", "")
    return directory_bytes(Path(cache_root) / "condensed" / name)
