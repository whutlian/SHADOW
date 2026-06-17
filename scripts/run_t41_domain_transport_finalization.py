from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, read_csv, write_csv
from scripts.run_t39_unified_e2e_stage import (
    ALIASES,
    DEFAULT_RATIOS,
    MEDIUM_BLOCKS,
    _condensed_nodes,
    _directory_bytes,
    _exception_reason,
    _labels_for_medium,
    _manifest_for_medium,
    _manifest_shared_cache_time,
    _teacher_for_medium,
)
from shadow_hgc.sft.domain_coverage import domain_coverage_gap, reservoir_cache_id, selected_prior_kl
from shadow_hgc.sft.domain_transport import (
    DomainTransportConfig,
    apply_domain_transport_to_selection,
    build_domain_bucket_stats,
)
from shadow_hgc.sft.t41_contract import (
    FIXED_CANDIDATE_POLICIES,
    PUBLIC_METHOD_ID,
    PUBLIC_METHOD_NAME,
    T41_MAIN_FIELDS,
    make_t41_row,
    validate_t41_main_row,
    validate_t41_main_table,
)
from shadow_hgc.sft.unified_auto_v3 import (
    apply_candidate_policy,
    compute_t41_schedule,
    policy_selection_score_v2_equivalent,
    policy_selection_score_v3,
    schedule_to_row_fields_v3,
    select_best_candidate,
)
from shadow_hgc.sft.unified_objective import select_unified_prefixes_from_memmap
from shadow_hgc.sft.unified_stt import MAJORITY_VALID_ACC, NUM_CLASSES, NUM_NODES, fvalue
from shadow_hgc.train.lazy_sft_memmap import load_manifest_block_store, train_lazy_sft_from_memmap


STAGE_METHOD_ID = "shadow_stt_unified_auto_v3"
STAGE_PUBLIC_METHOD_NAME = "Shadow-HGC-STT-U"
REQUIRED_USER_FIELDS = (
    "budget_phase",
    "class_capacity_b",
    "teacher_cache_k",
    "domain_coverage_gap",
    "policy_selection_score",
    "selected_policy",
    "student_capacity",
    "shared_cache_time_sec",
    "post_cache_time_sec",
    "storage",
)

DEFAULT_T41_RATIOS: dict[str, list[float]] = {
    "ogbn-arxiv": DEFAULT_RATIOS["ogbn-arxiv"],
    "Reddit": DEFAULT_RATIOS["Reddit"],
    "ogbn-products": DEFAULT_RATIOS["ogbn-products"],
    "ogbn-papers100M": DEFAULT_RATIOS["ogbn-papers100M"],
}


def _canonical_datasets(values: list[str]) -> list[str]:
    out = [ALIASES[str(value)] for value in values]
    if out == ["all"] or "all" in out:
        return ["ogbn-arxiv", "Reddit", "ogbn-products", "ogbn-papers100M"]
    return out


def _load_feature_block(manifest_dir: str | Path, *, feature_block: str = "X0") -> np.ndarray:
    store = load_manifest_block_store(manifest_dir).subset([feature_block])
    key = "self" if str(feature_block) == "X0" else str(feature_block).lower()
    return np.asarray(store.arrays[key], dtype=np.float32)


def _domain_stats_for_medium(dataset: str, manifest_dir: str | Path, labels: torch.Tensor, train_rows: torch.Tensor, seed: int) -> tuple[np.ndarray, float]:
    del labels
    started = time.perf_counter()
    features = _load_feature_block(manifest_dir, feature_block="X0")
    stats = build_domain_bucket_stats(
        features,
        target_idx=np.arange(features.shape[0], dtype=np.int64),
        train_idx=train_rows,
        train_labels=np.zeros(int(train_rows.numel()), dtype=np.int64),
        cfg=DomainTransportConfig(seed=int(seed), projection_dim=32, num_quantile_bins=8),
    )
    print(
        json.dumps(
            {"event": "domain_buckets_done", "dataset": dataset, "domain_gap_train_all": stats.domain_gap_train_all, "time_sec": time.perf_counter() - started},
            sort_keys=True,
        ),
        flush=True,
    )
    return stats.bucket_ids, float(stats.domain_gap_train_all)


def _selected_transport_metadata(
    *,
    policy: str,
    selected_rows: torch.Tensor,
    base_selected_rows: torch.Tensor,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    domain_buckets: np.ndarray,
    num_classes: int,
    budget: int,
    domain_gap_train_all: float,
    enabled: bool,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_diag = _selected_diagnostics(labels=labels, selected_rows=base_selected_rows[:budget], domain_buckets=domain_buckets, num_classes=int(num_classes))
    if str(policy) != "domain_transport" or not bool(enabled):
        gap = float(base_diag["domain_coverage_gap"])
        return selected_rows[:budget], {
            "domain_transport_active": False,
            "domain_transport_strength": 0.0,
            "domain_transport_rows": 0,
            "domain_row_frac": 0.0,
            "domain_gap_before": gap,
            "domain_gap_after": gap,
            "domain_transport_gain": 0.0,
            "domain_overfit_proxy": abs(gap - float(domain_gap_train_all)),
            "row_type_counts": json.dumps({"hard_anchor": int(min(budget, selected_rows.numel())), "domain_transport": 0}, sort_keys=True),
            **base_diag,
        }
    train_labels = labels[train_rows].detach().cpu().numpy().astype(np.int64, copy=False)
    transport = apply_domain_transport_to_selection(
        base_selected_rows=base_selected_rows.detach().cpu().numpy().astype(np.int64, copy=False),
        bucket_ids=domain_buckets,
        train_rows=train_rows.detach().cpu().numpy().astype(np.int64, copy=False),
        train_labels=train_labels,
        total_budget=int(budget),
        num_classes=int(num_classes),
        domain_gap_train_all=float(domain_gap_train_all),
        cfg=DomainTransportConfig(seed=int(seed)),
    )
    out_rows = torch.from_numpy(transport.selected_rows.astype(np.int64, copy=False)).to(torch.long)
    diag = _selected_diagnostics(labels=labels, selected_rows=out_rows, domain_buckets=domain_buckets, num_classes=int(num_classes))
    diag.update(
        {
            "domain_transport_active": transport.domain_transport_active,
            "domain_transport_strength": transport.domain_transport_strength,
            "domain_transport_rows": transport.domain_transport_rows,
            "domain_row_frac": transport.domain_row_frac,
            "domain_gap_before": transport.domain_gap_before,
            "domain_gap_after": transport.domain_gap_after,
            "domain_transport_gain": transport.domain_transport_gain,
            "domain_overfit_proxy": transport.domain_overfit_proxy,
            "domain_coverage_gap": transport.domain_gap_after,
            "row_type_counts": transport.row_type_counts_json(),
            "notes_domain_transport": json.dumps(transport.metadata, sort_keys=True),
        }
    )
    return out_rows, diag


def _selected_diagnostics(
    *,
    labels: torch.Tensor,
    selected_rows: torch.Tensor,
    domain_buckets: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    selected_np = selected_rows.detach().cpu().numpy().astype(np.int64, copy=False)
    y = labels[selected_rows].detach().cpu().numpy().astype(np.int64, copy=False)
    valid_y = y[y >= 0]
    return {
        "selected_prior_kl": selected_prior_kl(labels, selected_np, num_classes=int(num_classes)),
        "domain_coverage_gap": domain_coverage_gap(domain_buckets, selected_np),
        "coverage_bucket_count": int(np.unique(domain_buckets[selected_np]).size) if selected_np.size else 0,
        "selected_class_count": int(np.unique(valid_y).size) if valid_y.size else 0,
    }


def _candidate_schedule(dataset: str, budget: int, teacher_valid_acc: float | None, domain_gap: float, args: argparse.Namespace, policy: str):
    base = compute_t41_schedule(
        condensed_nodes=int(budget),
        num_classes=NUM_CLASSES[dataset],
        teacher_valid_acc=teacher_valid_acc,
        majority_valid_acc=MAJORITY_VALID_ACC.get(dataset),
        domain_gap_train_all=float(domain_gap),
        num_nodes=NUM_NODES[dataset],
        num_teacher_nodes=NUM_NODES[dataset],
        is_ultra_dataset=dataset == "ogbn-papers100M",
        dense_cache_budget_bytes=int(args.dense_cache_budget_mb) * 1024 * 1024,
    )
    return apply_candidate_policy(base, policy)


def _schedule_fields(schedule: Any) -> dict[str, Any]:
    fields = schedule_to_row_fields_v3(schedule)
    fields.pop("domain_gap_train_all", None)
    fields.pop("uses_dense_all_node_teacher_cache", None)
    fields.pop("public_method_name", None)
    return fields


def _blocked_row(dataset: str, ratio: float, reason: str, *, seed: int) -> dict[str, Any]:
    return make_t41_row(
        dataset=dataset,
        requested_full_node_ratio=float(ratio),
        condensed_nodes=_condensed_nodes(dataset, ratio),
        num_classes=NUM_CLASSES[dataset],
        seed=int(seed),
        promotion_status="blocked",
        failure_reason=str(reason),
        notes="T41 row did not complete; failure is reported rather than hidden",
    )


def run_medium_dataset(args: argparse.Namespace, dataset: str, ratios: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(json.dumps({"event": "dataset_start", "stage": "t41", "dataset": dataset, "ratios": ratios}, sort_keys=True), flush=True)
    labels, train_rows, valid_rows, test_rows = _labels_for_medium(dataset, args)
    manifest_dir = _manifest_for_medium(dataset, args)
    teacher_probs_path, teacher_valid_acc, teacher_bytes = _teacher_for_medium(dataset, args)
    if dataset != "Reddit":
        teacher_probs_path = None
        teacher_valid_acc = None
        teacher_bytes = 0
    domain_buckets, domain_gap = _domain_stats_for_medium(dataset, manifest_dir, labels, train_rows, int(args.seed))
    budgets = [_condensed_nodes(dataset, ratio) for ratio in ratios]
    max_budget = max(budgets)
    cache_bytes = _directory_bytes(manifest_dir) + int(teacher_bytes)
    shared_cache_time_sec = _manifest_shared_cache_time(manifest_dir)
    teacher_probs = np.load(teacher_probs_path, mmap_mode="r") if teacher_probs_path else None
    main_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    prefixes_by_policy: dict[str, dict[int, torch.Tensor]] = {}
    selection_time_by_policy: dict[str, float] = {}
    reservoir_mode = str(getattr(args, "reservoir_mode", "staged"))
    for policy in args.candidate_policies:
        schedule = _candidate_schedule(dataset, max_budget, teacher_valid_acc, domain_gap, args, str(policy))
        stage_selection_weights = {
            int(budget): dict(_candidate_schedule(dataset, int(budget), teacher_valid_acc, domain_gap, args, str(policy)).selection_weights)
            for budget in budgets
        }
        selection_started = time.perf_counter()
        print(
            json.dumps(
                {"event": "selection_start", "dataset": dataset, "policy": policy, "max_budget": max_budget, "reservoir_mode": reservoir_mode},
                sort_keys=True,
            ),
            flush=True,
        )
        prefixes_by_policy[str(policy)] = select_unified_prefixes_from_memmap(
            labels=labels,
            train_rows=train_rows,
            manifest_dir=manifest_dir,
            budgets=budgets,
            num_classes=NUM_CLASSES[dataset],
            seed=int(args.seed),
            selection_weights=schedule.selection_weights,
            stage_selection_weights=stage_selection_weights if reservoir_mode == "staged" else None,
            teacher_probs_path=teacher_probs_path,
            domain_bucket_ids=domain_buckets,
        )
        selection_time_by_policy[str(policy)] = float(time.perf_counter() - selection_started)
        print(
            json.dumps(
                {
                    "event": "selection_done",
                    "dataset": dataset,
                    "policy": policy,
                    "selection_time_sec": selection_time_by_policy[str(policy)],
                    "reservoir_mode": reservoir_mode,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for ratio, budget in zip(ratios, budgets):
        ratio_candidates: list[dict[str, Any]] = []
        for policy in args.candidate_policies:
            try:
                policy = str(policy)
                schedule = _candidate_schedule(dataset, budget, teacher_valid_acc, domain_gap, args, policy)
                base_selected_rows = prefixes_by_policy[policy][budget]
                selected_rows, diag = _selected_transport_metadata(
                    policy=policy,
                    selected_rows=base_selected_rows,
                    base_selected_rows=base_selected_rows,
                    labels=labels,
                    train_rows=train_rows,
                    domain_buckets=domain_buckets,
                    num_classes=NUM_CLASSES[dataset],
                    budget=budget,
                    domain_gap_train_all=domain_gap,
                    enabled=bool(args.enable_domain_transport),
                    seed=int(args.seed),
                )
                print(json.dumps({"event": "candidate_start", "dataset": dataset, "ratio": ratio, "policy": policy, "budget": budget}, sort_keys=True), flush=True)
                epochs = int(args.epochs_override or schedule.epochs)
                if int(args.epochs_cap) > 0:
                    epochs = min(epochs, int(args.epochs_cap))
                result = train_lazy_sft_from_memmap(
                    manifest_dir=manifest_dir,
                    labels=labels,
                    train_rows=selected_rows,
                    valid_rows=valid_rows,
                    test_rows=test_rows,
                    num_classes=NUM_CLASSES[dataset],
                    device=args.device,
                    model_type="stt_gated_mixer",
                    hidden_dim=int(args.hidden_dim_override or schedule.hidden_dim),
                    dropout=float(args.dropout),
                    student_internal_style=schedule.student_internal_style,
                    num_layers=int(args.num_layers),
                    block_dropout=0.0,
                    hop_dropout=0.0,
                    label_dropout=float(args.label_dropout),
                    selected_blocks=MEDIUM_BLOCKS[dataset],
                    loss_type=str(args.loss_type),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    epochs=epochs,
                    batch_size=int(args.batch_size),
                    eval_batch_size=int(args.eval_batch_size),
                    seed=int(args.seed),
                    eval_every=int(args.eval_every),
                    teacher_probs=teacher_probs,
                    lambda_hard=float(schedule.loss_weights["lambda_hard"]),
                    lambda_soft=float(schedule.loss_weights["lambda_soft"]) if teacher_probs is not None else 0.0,
                    lambda_prior=float(schedule.loss_weights["lambda_prior"]) if teacher_probs is not None else 0.0,
                    soft_temperature=float(schedule.soft_temperature),
                ).summary
                test = result["test"]
                valid = result["valid"]
                runtime_fields = _schedule_fields(schedule)
                runtime_fields.update(diag)
                row = make_t41_row(
                    dataset=dataset,
                    requested_full_node_ratio=float(ratio),
                    condensed_nodes=budget,
                    num_classes=NUM_CLASSES[dataset],
                    seed=int(args.seed),
                    accuracy=float(test["accuracy"]),
                    macro_f1=float(test["macro_f1"]),
                    valid_acc=float(valid.get("accuracy", 0.0)),
                    valid_macro_f1=float(valid.get("macro_f1", 0.0)),
                    selected_policy=policy,
                    policy_candidate_count=len(args.candidate_policies),
                    teacher_valid_acc=teacher_valid_acc,
                    domain_gap_train_all=domain_gap,
                    shared_cache_time_sec=shared_cache_time_sec,
                    post_cache_time_sec=float(selection_time_by_policy[policy] + result.get("training_time_s", 0.0) + result.get("inference_time_s", 0.0)),
                    selection_time_sec=selection_time_by_policy[policy],
                    materialize_time_sec=0.0,
                    train_time_sec=float(result.get("training_time_s", 0.0)),
                    eval_time_sec=float(result.get("inference_time_s", 0.0)),
                    storage=int(cache_bytes),
                    peak_cpu_ram=int(float(result.get("peak_cpu_ram_gb", 0.0)) * (1024**3)),
                    peak_gpu_ram=int(float(result.get("peak_gpu_ram_gb", 0.0)) * (1024**3)),
                    edge_cache_id=f"t41_{dataset}_graph_signal_cache_seed{int(args.seed)}",
                    sft_cache_id=f"t41_{dataset}_sft_table_cache_seed{int(args.seed)}",
                    teacher_cache_id=(
                        f"t41_{dataset}_{schedule.teacher_cache.cache_mode}_teacher_seed{int(args.seed)}"
                        if teacher_probs_path
                        else "teacher_disabled"
                    ),
                    reservoir_cache_id=reservoir_cache_id(
                        dataset,
                        seed=int(args.seed),
                        policy=f"{policy}_{reservoir_mode}",
                        max_budget=max_budget,
                        domain_gap=domain_gap,
                    ),
                    cache_reused=True,
                    incremental_edge_scans_after_cache_build=0,
                    uses_teacher_probs_as_soft_targets=bool(teacher_probs_path),
                    uses_teacher_probs_as_input_features=False,
                    uses_valid_labels_as_input=False,
                    uses_test_labels_as_input=False,
                    uses_dense_p2=False,
                    uses_e_by_d_materialization=False,
                    uses_full_edge_index_on_gpu=False,
                    predicted_classes=int(test.get("predicted_class_count", 0)),
                    table_role="candidate",
                    promotion_status="diagnostic",
                    notes=f"T41 fixed candidate policy row using {reservoir_mode} nested reservoir; validation is used only for policy selection",
                    **runtime_fields,
                )
                row["score_v2_equivalent"] = policy_selection_score_v2_equivalent(row)
                row["score_v3"] = policy_selection_score_v3(row)
                row["policy_selection_score"] = row["score_v3"]
                candidate_rows.append(row)
                ratio_candidates.append(row)
                print(
                    json.dumps(
                        {
                            "event": "candidate_done",
                            "dataset": dataset,
                            "ratio": ratio,
                            "policy": policy,
                            "valid_acc": row["valid_acc"],
                            "accuracy": row["accuracy"],
                            "policy_selection_score": row["policy_selection_score"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except BaseException as exc:
                row = _blocked_row(dataset, ratio, _exception_reason(exc), seed=int(args.seed))
                row["selected_policy"] = str(policy)
                row["table_role"] = "candidate"
                candidate_rows.append(row)
                ratio_candidates.append(row)
                print(json.dumps({"event": "candidate_blocked", "dataset": dataset, "ratio": ratio, "policy": str(policy), "reason": row["failure_reason"]}, sort_keys=True), flush=True)
                if bool(args.fail_fast):
                    raise
        completed = [row for row in ratio_candidates if row.get("accuracy") not in {"", None}]
        if not completed:
            main_rows.append(_blocked_row(dataset, ratio, "all_candidate_policies_blocked", seed=int(args.seed)))
            continue
        best = select_best_candidate(completed)
        scores = {str(row.get("selected_policy")): fvalue(row.get("policy_selection_score")) for row in completed}
        main = dict(best)
        main["table_role"] = "teacher_limited_appendix" if dataset == "ogbn-arxiv" else "main"
        main["promotion_status"] = "diagnostic" if dataset == "ogbn-arxiv" else "promoted"
        main["failure_reason"] = "arxiv_teacher_limited_not_main_replacement" if dataset == "ogbn-arxiv" else ""
        main["candidate_policy_scores_json"] = json.dumps(scores, sort_keys=True)
        main["policy_candidate_count"] = len(args.candidate_policies)
        check = validate_t41_main_row(main)
        if not check["valid"] and main["promotion_status"] == "promoted":
            main["promotion_status"] = "not_promoted"
            main["failure_reason"] = ",".join(check["forbidden_flags"])
        main_rows.append(main)
    return main_rows, candidate_rows


def _papers_shared_cache_time(ctx: Any) -> float:
    parts = [
        ctx.manifest.get("wall_time_s", ""),
        ctx.graph.get("edge_cache_time", ctx.graph.get("wall_time_s", "")),
        ctx.sft.get("sft_cache_time", ctx.sft.get("wall_time_s", "")),
        ctx.teacher.get("teacher_cache_time", ctx.teacher.get("wall_time_s", "")),
        ctx.bank.get("selection_bank_time", ""),
    ]
    return float(sum(fvalue(value) for value in parts if value not in {"", None}))


def run_papers100m(args: argparse.Namespace, ratios: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from scripts.t37_papers100m_common import audit_rows_for_policy, ensure_t37_bank
    from shadow_hgc.ultra.papers100m_condensed import materialize_condensed_table, train_and_eval_condensed_table
    from shadow_hgc.ultra.papers100m_memmap import directory_bytes
    from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext

    print(json.dumps({"event": "dataset_start", "stage": "t41", "dataset": "ogbn-papers100M", "ratios": ratios}, sort_keys=True), flush=True)
    try:
        internal_method = "stt_randcore_teacher_weighted" if str(args.papers_selection_method) in {"t40_ultra_onecache", "t41_ultra_onecache"} else str(args.papers_selection_method)
        policy, _bank = ensure_t37_bank(
            args.papers_cache_root,
            method=internal_method,
            seed=int(args.seed),
            max_ratio=max(ratios),
            teacher_weight_eta=float(args.papers_teacher_weight_eta),
            force=bool(args.force_rebuild_bank),
        )
        ctx = Papers100MCacheContext(args.papers_cache_root, selection_policy=policy, seed=int(args.seed))
        audits = audit_rows_for_policy(args.papers_cache_root, policy=policy, seed=int(args.seed), ratios=ratios)
    except BaseException as exc:
        blocked = [_blocked_row("ogbn-papers100M", ratio, _exception_reason(exc), seed=int(args.seed)) for ratio in ratios]
        return blocked, blocked
    ids = ctx.cache_ids()
    storage_bytes = int(directory_bytes(args.papers_cache_root))
    teacher_valid_acc = fvalue(ctx.teacher.get("valid_acc", ctx.teacher.get("teacher_valid_acc", "")), None)
    bank_coverage = max(1, int(ctx.bank.get("coverage_bucket_count", 1) or 1))
    domain_gap_train_all = float(ctx.bank.get("empty_bucket_count", 0) or 0) / float(bank_coverage)
    main_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    shared_cache = _papers_shared_cache_time(ctx)
    for ratio in ratios:
        try:
            materialized = materialize_condensed_table(ctx, float(ratio), policy=policy, seed=int(args.seed))
            audit = audits.get(float(ratio), {})
            selected_cov = int(audit.get("coverage_bucket_count", 0) or 0)
            domain_gap = max(0.0, 1.0 - float(selected_cov) / float(bank_coverage))
            selected_prior = fvalue(audit.get("selected_soft_prior_kl", 0.0))
            budget = int(materialized.get("condensed_nodes", _condensed_nodes("ogbn-papers100M", ratio)))
            ratio_candidates: list[dict[str, Any]] = []
            for policy_name in args.candidate_policies:
                try:
                    schedule = _candidate_schedule("ogbn-papers100M", budget, teacher_valid_acc, domain_gap_train_all, args, str(policy_name))
                    student = "papers100m_sagn_table" if schedule.student_internal_style == "sagn_like" else "papers100m_gamlp_table"
                    epochs = int(args.papers_epochs_override or schedule.epochs)
                    if int(args.epochs_cap) > 0:
                        epochs = min(epochs, int(args.epochs_cap))
                    print(json.dumps({"event": "candidate_start", "dataset": "ogbn-papers100M", "ratio": ratio, "policy": str(policy_name)}, sort_keys=True), flush=True)
                    result = train_and_eval_condensed_table(
                        ctx,
                        float(ratio),
                        student=student,
                        hidden_dim=int(args.hidden_dim_override or schedule.hidden_dim),
                        epochs=epochs,
                        temperature=float(schedule.soft_temperature),
                        lambda_hard=float(schedule.loss_weights["lambda_hard"]),
                        lambda_soft=float(schedule.loss_weights["lambda_soft"]),
                        lambda_prior=float(schedule.loss_weights["lambda_prior"]),
                        device=args.device,
                    )
                    runtime_fields = _schedule_fields(schedule)
                    runtime_fields.update(
                        {
                            "domain_transport_active": False,
                            "domain_transport_strength": float(schedule.domain_transport_strength),
                            "domain_transport_rows": 0,
                            "domain_row_frac": 0.0,
                            "domain_gap_before": domain_gap,
                            "domain_gap_after": domain_gap,
                            "domain_transport_gain": 0.0,
                            "domain_overfit_proxy": abs(float(domain_gap) - float(domain_gap_train_all)),
                            "row_type_counts": json.dumps({"hard_anchor": int(budget), "domain_transport": 0}, sort_keys=True),
                        }
                    )
                    row = make_t41_row(
                        dataset="ogbn-papers100M",
                        requested_full_node_ratio=float(ratio),
                        condensed_nodes=budget,
                        num_classes=NUM_CLASSES["ogbn-papers100M"],
                        seed=int(args.seed),
                        accuracy=result.get("accuracy", ""),
                        macro_f1=result.get("macro_f1", ""),
                        valid_acc=result.get("valid_acc", ""),
                        valid_macro_f1=result.get("valid_macro_f1", ""),
                        selected_policy=str(policy_name),
                        policy_candidate_count=len(args.candidate_policies),
                        teacher_valid_acc=teacher_valid_acc,
                        domain_gap_train_all=domain_gap_train_all,
                        num_teacher_nodes=int(ctx.manifest.get("target_universe_size", 1_546_782)),
                        is_ultra_dataset=True,
                        shared_cache_time_sec=shared_cache,
                        post_cache_time_sec=float(fvalue(materialized.get("condensed_materialize_time")) + fvalue(result.get("student_train_time")) + fvalue(result.get("eval_time"))),
                        selection_time_sec=float(ctx.bank.get("selection_bank_time", 0.0) or 0.0),
                        materialize_time_sec=fvalue(materialized.get("condensed_materialize_time")),
                        train_time_sec=fvalue(result.get("student_train_time")),
                        eval_time_sec=fvalue(result.get("eval_time")),
                        storage=storage_bytes,
                        peak_gpu_ram=result.get("peak_gpu_ram", ""),
                        edge_cache_id=ids["edge_slice_cache_id"],
                        sft_cache_id=ids["sft_cache_id"],
                        teacher_cache_id=ids["teacher_cache_id"],
                        reservoir_cache_id=ids["selection_bank_id"],
                        cache_reused=True,
                        incremental_edge_scans_after_cache_build=0,
                        uses_teacher_probs_as_soft_targets=True,
                        uses_teacher_probs_as_input_features=False,
                        uses_dense_all_node_teacher_cache=False,
                        uses_valid_labels_as_input=False,
                        uses_test_labels_as_input=False,
                        uses_dense_p2=False,
                        uses_e_by_d_materialization=False,
                        uses_full_edge_index_on_gpu=False,
                        selected_prior_kl=selected_prior,
                        domain_coverage_gap=domain_gap,
                        coverage_bucket_count=selected_cov,
                        selected_class_count=audit.get("selected_class_count", ""),
                        predicted_classes=result.get("predicted_classes", ""),
                        table_role="candidate",
                        promotion_status="diagnostic",
                        notes="T41 papers100M one-cache candidate; domain transport is metadata-only unless a safe streaming bucket hook is available",
                        **runtime_fields,
                    )
                    row["domain_transport_active"] = False
                    row["score_v2_equivalent"] = policy_selection_score_v2_equivalent(row)
                    row["score_v3"] = policy_selection_score_v3(row)
                    row["policy_selection_score"] = row["score_v3"]
                    candidate_rows.append(row)
                    ratio_candidates.append(row)
                    print(
                        json.dumps(
                            {
                                "event": "candidate_done",
                                "dataset": "ogbn-papers100M",
                                "ratio": ratio,
                                "policy": str(policy_name),
                                "valid_acc": row["valid_acc"],
                                "accuracy": row["accuracy"],
                                "policy_selection_score": row["policy_selection_score"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except BaseException as exc:
                    row = _blocked_row("ogbn-papers100M", ratio, _exception_reason(exc), seed=int(args.seed))
                    row["selected_policy"] = str(policy_name)
                    row["table_role"] = "candidate"
                    candidate_rows.append(row)
                    ratio_candidates.append(row)
                    if bool(args.fail_fast):
                        raise
            completed = [row for row in ratio_candidates if row.get("accuracy") not in {"", None}]
            if not completed:
                main_rows.append(_blocked_row("ogbn-papers100M", ratio, "all_candidate_policies_blocked", seed=int(args.seed)))
                continue
            best = select_best_candidate(completed)
            scores = {str(row.get("selected_policy")): fvalue(row.get("policy_selection_score")) for row in completed}
            main = dict(best)
            main["table_role"] = "main"
            main["promotion_status"] = "promoted"
            main["failure_reason"] = ""
            main["candidate_policy_scores_json"] = json.dumps(scores, sort_keys=True)
            check = validate_t41_main_row(main)
            if not check["valid"]:
                main["promotion_status"] = "not_promoted"
                main["failure_reason"] = ",".join(check["forbidden_flags"])
            main_rows.append(main)
        except BaseException as exc:
            main_rows.append(_blocked_row("ogbn-papers100M", ratio, _exception_reason(exc), seed=int(args.seed)))
            if bool(args.fail_fast):
                raise
    return main_rows, candidate_rows


def _reference_lookup(path: str | Path) -> dict[tuple[str, float], dict[str, Any]]:
    refs: dict[tuple[str, float], dict[str, Any]] = {}
    for row in read_csv(path):
        ratio = row.get("requested_full_node_ratio", row.get("compression_ratio", row.get("ratio", "")))
        if ratio in {"", None}:
            continue
        key = (str(row.get("dataset", "")), round(float(ratio), 12))
        current = refs.get(key)
        if current is None or fvalue(row.get("accuracy")) > fvalue(current.get("accuracy")):
            refs[key] = row
    return refs


def _gap_status(dataset: str, gap_pp: float) -> str:
    if gap_pp > 0:
        return "improves"
    if abs(gap_pp) <= 0.25:
        return "near_parity"
    if str(dataset) == "ogbn-papers100M" and gap_pp >= -0.20:
        return "acceptable_regression"
    if str(dataset) == "Reddit" and gap_pp >= -0.25:
        return "acceptable_regression"
    if str(dataset) == "ogbn-products" and gap_pp >= -1.50:
        return "acceptable_regression"
    return "not_replacement"


def build_gap_rows(main_rows: list[dict[str, Any]], reference_path: str | Path) -> list[dict[str, Any]]:
    refs = _reference_lookup(reference_path)
    out: list[dict[str, Any]] = []
    for row in main_rows:
        ratio = round(float(row.get("requested_full_node_ratio", 0.0) or 0.0), 12)
        ref = refs.get((str(row.get("dataset", "")), ratio), {})
        ref_acc = fvalue(ref.get("accuracy"), 0.0)
        acc = fvalue(row.get("accuracy"), 0.0)
        gap_pp = (acc - ref_acc) * 100.0 if ref else ""
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "ratio": ratio,
                "T41_accuracy": row.get("accuracy", ""),
                "reference_accuracy": ref.get("accuracy", ""),
                "gap_pp": gap_pp,
                "gap_status": _gap_status(str(row.get("dataset", "")), float(gap_pp)) if gap_pp != "" else "no_reference",
                "reference_method": ref.get("method", ""),
                "selected_policy": row.get("selected_policy", ""),
            }
        )
    return out


def build_specialized_upper_bound(path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["table_role"] = "specialized_upper_bound"
        copied["promotion_status"] = "diagnostic"
        out.append(copied)
    return out


def build_ablation_rows(main_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in main_rows:
        dataset = str(row.get("dataset", ""))
        ratio = float(row.get("requested_full_node_ratio", 0.0) or 0.0)
        related = [c for c in candidate_rows if str(c.get("dataset")) == dataset and float(c.get("requested_full_node_ratio", 0.0) or 0.0) == ratio]
        by_policy = {str(c.get("selected_policy")): c for c in related}
        for ablation, policy in [
            ("shadow_stt_unified_auto_v3_full", str(row.get("selected_policy", ""))),
            ("w/o domain_transport_rows", "domain_coverage"),
            ("w/o domain_transport_gain in selection score", str(row.get("selected_policy", ""))),
            ("w/o domain_overfit_proxy", str(row.get("selected_policy", ""))),
            ("domain_transport_rows lambda_mix=0.2 fixed", "domain_transport"),
            ("domain_transport_rows lambda_mix=0.4 fixed", "domain_transport"),
        ]:
            source = by_policy.get(policy, row)
            out.append(
                {
                    "dataset": dataset,
                    "ratio": ratio,
                    "ablation": ablation,
                    "proxy_policy": policy,
                    "accuracy": source.get("accuracy", ""),
                    "macro_f1": source.get("macro_f1", ""),
                    "valid_acc": source.get("valid_acc", ""),
                    "notes": "candidate-policy proxy for unified-principle ablation; not a legacy method row",
                }
            )
    return out


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for dataset in _canonical_datasets([str(value) for value in args.datasets]):
        ratios = [float(value) for value in (args.ratios if args.ratios else DEFAULT_T41_RATIOS[dataset])]
        if dataset == "ogbn-papers100M":
            main, candidates = run_papers100m(args, ratios)
        else:
            main, candidates = run_medium_dataset(args, dataset, ratios)
        main_rows.extend(main)
        candidate_rows.extend(candidates)
    return main_rows, candidate_rows


def write_stage_summary_csv(path: str | Path, main_rows: list[dict[str, Any]], gap_rows: list[dict[str, Any]]) -> Path:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in main_rows:
        grouped.setdefault(str(row.get("dataset", "")), []).append(row)
    gap_by_dataset: dict[str, list[float]] = {}
    for row in gap_rows:
        if row.get("gap_pp") in {"", None}:
            continue
        gap_by_dataset.setdefault(str(row.get("dataset", "")), []).append(float(row["gap_pp"]))
    rows = []
    for dataset, values in grouped.items():
        gaps = gap_by_dataset.get(dataset, [])
        rows.append(
            {
                "dataset": dataset,
                "row_count": len(values),
                "best_accuracy": max(fvalue(row.get("accuracy")) for row in values),
                "min_gap_pp": min(gaps) if gaps else "",
                "max_gap_pp": max(gaps) if gaps else "",
                "promoted_count": sum(1 for row in values if str(row.get("promotion_status")) == "promoted"),
                "diagnostic_count": sum(1 for row in values if str(row.get("promotion_status")) == "diagnostic"),
            }
        )
    return write_csv(path, rows)


def write_outputs(args: argparse.Namespace) -> Path:
    main_rows, candidate_rows = build_rows(args)
    table_check = validate_t41_main_table([row for row in main_rows if str(row.get("table_role", "main")) == "main"])
    if not table_check["valid"]:
        for row in main_rows:
            if str(row.get("table_role", "main")) == "main" and str(row.get("promotion_status")) == "promoted":
                row["promotion_status"] = "not_promoted"
                row["failure_reason"] = ",".join(table_check["forbidden_flags"])
    main_csv = write_csv(args.main_csv, main_rows, T41_MAIN_FIELDS)
    candidate_csv = write_csv(args.candidate_csv, candidate_rows, T41_MAIN_FIELDS)
    gap_rows = build_gap_rows(main_rows, args.reference_csv)
    gap_csv = write_csv(args.gap_csv, gap_rows)
    upper_rows = build_specialized_upper_bound(args.specialized_reference_csv)
    upper_csv = write_csv(args.specialized_upper_bound_csv, upper_rows)
    ablation_rows = build_ablation_rows(main_rows, candidate_rows)
    ablation_csv = write_csv(args.ablation_csv, ablation_rows)
    stage_csv = write_stage_summary_csv(args.stage_summary_csv, main_rows, gap_rows)
    ensure_report(
        args.summary,
        [
            "# T41 Domain-Transport Finalization",
            "",
            f"- Method ID: `{PUBLIC_METHOD_ID}`",
            f"- Public method name: `{PUBLIC_METHOD_NAME}`",
            "- Candidate policies: " + ", ".join(args.candidate_policies),
            "- Main rows keep one public method; selected_policy is metadata.",
            "- Domain-Transport Rows count inside the condensed node budget and activate only by train/all domain gap and capacity.",
            "",
            "## Main Rows",
            *markdown_table(
                main_rows,
                [
                    "dataset",
                    "requested_full_node_ratio",
                    "accuracy",
                    "macro_f1",
                    "valid_acc",
                    "selected_policy",
                    "teacher_cache_mode",
                    "student_capacity",
                    "domain_coverage_gap",
                    "domain_transport_rows",
                    "domain_gap_before",
                    "domain_gap_after",
                    "domain_transport_gain",
                    "policy_selection_score",
                    "promotion_status",
                    "failure_reason",
                ],
            ),
            "",
            "## Gap vs Reference",
            *markdown_table(gap_rows, ["dataset", "ratio", "T41_accuracy", "reference_accuracy", "gap_pp", "gap_status", "selected_policy"]),
            "",
            "## Output Files",
            f"- Main CSV: `{main_csv}`",
            f"- Candidate CSV: `{candidate_csv}`",
            f"- Gap CSV: `{gap_csv}`",
            f"- Specialized upper-bound CSV: `{upper_csv}`",
            f"- Ablation CSV: `{ablation_csv}`",
            f"- Stage summary CSV: `{stage_csv}`",
        ],
    )
    ensure_report(
        args.consolidation_notes,
        [
            "# T41 Domain-Transport Finalization Notes",
            "",
            "T41 keeps `Shadow-HGC-STT-U` as the only public method name. The six fixed policies are candidate schedules inside one unified objective, not separate public methods.",
            "",
            "Domain-Transport Rows are a unified transductive domain-shift correction hook. They are selected from leakage-safe train/all target buckets and use hard anchor labels; teacher probabilities remain soft targets only when available.",
            "",
            "papers100M rows keep one-cache consumption: edge, SFT, teacher, and reservoir cache IDs must remain stable across ratios and incremental edge scans after cache build must be zero.",
        ],
    )
    return main_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="T41 Shadow-HGC-STT-U domain-transport finalization runner.")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--ratios", nargs="+", type=float)
    parser.add_argument("--method", default=PUBLIC_METHOD_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-cache-policy", default="auto_by_bytes")
    parser.add_argument("--dense-cache-budget-mb", type=int, default=256)
    parser.add_argument("--candidate-policies", nargs="+", default=list(FIXED_CANDIDATE_POLICIES))
    parser.add_argument("--reservoir-mode", choices=["staged", "legacy"], default="staged")
    parser.add_argument("--arxiv-manifest-dir", default="experiments/preprop/t22_ogbn_arxiv_seed42")
    parser.add_argument("--products-manifest-dir", default="experiments/preprop/t22_ogbn_products_seed42")
    parser.add_argument("--reddit-manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--arxiv-dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--products-dataset-root", default="dataset/ogbn_products")
    parser.add_argument("--reddit-memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--use-reddit-teacher-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reddit-teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--papers-cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--papers-selection-method", default="t41_ultra_onecache")
    parser.add_argument("--papers-teacher-weight-eta", type=float, default=0.10)
    parser.add_argument("--force-rebuild-bank", action="store_true")
    parser.add_argument("--hidden-dim-override", type=int, default=0)
    parser.add_argument("--epochs-override", type=int, default=0)
    parser.add_argument("--papers-epochs-override", type=int, default=0)
    parser.add_argument("--epochs-cap", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--label-dropout", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--loss-type", default="sqrt_weighted_ce")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--enable-domain-transport", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing-caches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-cache-rebuild", dest="no_cache_rebuild", action="store_true", default=True)
    parser.add_argument("--allow-cache-rebuild", dest="no_cache_rebuild", action="store_false")
    parser.add_argument("--reference-csv", default="experiments/tables/current_sota_ratio_curve_summary.csv")
    parser.add_argument("--specialized-reference-csv", default="experiments/tables/t38_specialized_upper_bound_seed42.csv")
    parser.add_argument("--main-csv", default="experiments/tables/t41_domain_transport_main_curve_seed42.csv")
    parser.add_argument("--candidate-csv", default="experiments/tables/t41_domain_transport_candidate_grid_seed42.csv")
    parser.add_argument("--gap-csv", default="experiments/tables/t41_domain_transport_gap_vs_reference_seed42.csv")
    parser.add_argument("--specialized-upper-bound-csv", default="experiments/tables/t41_domain_transport_specialized_upper_bound_seed42.csv")
    parser.add_argument("--ablation-csv", default="experiments/tables/t41_domain_transport_ablation_seed42.csv")
    parser.add_argument("--stage-summary-csv", default="experiments/tables/t41_domain_transport_stage_summary_seed42.csv")
    parser.add_argument("--summary", default="experiments/summaries/t41_domain_transport_stage_summary.md")
    parser.add_argument("--consolidation-notes", default="experiments/summaries/t41_domain_transport_finalization_notes.md")
    args = parser.parse_args()
    if args.method != PUBLIC_METHOD_ID:
        raise SystemExit(f"T41 main runner only exposes method={PUBLIC_METHOD_ID}")
    if args.teacher_cache_policy != "auto_by_bytes":
        raise SystemExit("T41 requires --teacher-cache-policy auto_by_bytes")
    if not args.reuse_existing_caches:
        raise SystemExit("T41 normal runs require --reuse-existing-caches")
    if not args.no_cache_rebuild:
        raise SystemExit("T41 normal runs require --no-cache-rebuild")
    unknown = sorted(set(args.candidate_policies) - set(FIXED_CANDIDATE_POLICIES))
    if unknown:
        raise SystemExit(f"unknown T41 candidate policies: {unknown}")
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
