from __future__ import annotations

from shadow_hgc.sft.t41_contract import (
    FIXED_CANDIDATE_POLICIES,
    PUBLIC_METHOD_ID,
    PUBLIC_METHOD_NAME,
    make_t41_row,
    validate_t41_main_row,
    validate_t41_main_table,
)
from shadow_hgc.sft.unified_auto_v3 import (
    apply_candidate_policy,
    compute_t41_schedule,
    policy_selection_score_v2_equivalent,
    policy_selection_score_v3,
    select_best_candidate,
)


def test_t41_candidate_policy_set_adds_domain_transport_without_new_public_name() -> None:
    assert PUBLIC_METHOD_ID == "shadow_stt_unified_auto_v3"
    assert PUBLIC_METHOD_NAME == "Shadow-HGC-STT-U"
    assert FIXED_CANDIDATE_POLICIES == (
        "auto_base",
        "coverage_heavy",
        "domain_coverage",
        "teacher_transport",
        "high_fidelity",
        "domain_transport",
    )

    base = compute_t41_schedule(
        condensed_nodes=12_245,
        num_classes=47,
        teacher_valid_acc=0.9,
        majority_valid_acc=0.09,
        domain_gap_train_all=0.28,
        num_nodes=2_449_029,
    )
    domain = apply_candidate_policy(base, "domain_transport")

    assert domain.policy_name == "domain_transport"
    assert domain.public_method_name == PUBLIC_METHOD_NAME
    assert domain.domain_transport_strength >= 0.25
    assert domain.domain_transport_active is True
    assert domain.selection_weights["domain"] >= base.selection_weights["domain"]
    assert domain.loss_weights["lambda_domain"] >= base.loss_weights["lambda_domain"]


def test_t41_selection_score_uses_transport_gain_and_overfit_proxy() -> None:
    rows = [
        {
            "selected_policy": "domain_coverage",
            "valid_acc": 0.900,
            "valid_macro_f1": 0.50,
            "selected_prior_kl": 0.0,
            "domain_coverage_gap": 0.18,
            "domain_transport_gain": 0.00,
            "domain_overfit_proxy": 0.01,
        },
        {
            "selected_policy": "domain_transport",
            "valid_acc": 0.8997,
            "valid_macro_f1": 0.50,
            "selected_prior_kl": 0.0,
            "domain_coverage_gap": 0.08,
            "domain_transport_gain": 0.10,
            "domain_overfit_proxy": 0.02,
        },
    ]

    assert policy_selection_score_v2_equivalent(rows[0]) > 0.0
    assert policy_selection_score_v3(rows[1]) > policy_selection_score_v3(rows[0])
    assert select_best_candidate(rows)["selected_policy"] == "domain_transport"


def test_t41_row_contract_requires_new_fields_and_guards_legacy_ids() -> None:
    row = make_t41_row(
        dataset="ogbn-products",
        requested_full_node_ratio=0.0025,
        condensed_nodes=6123,
        num_classes=47,
        accuracy=0.74,
        macro_f1=0.61,
        valid_acc=0.75,
        valid_macro_f1=0.62,
        selected_policy="domain_transport",
        promotion_status="promoted",
        domain_gap_train_all=0.25,
        domain_gap_before=0.20,
        domain_gap_after=0.08,
        domain_transport_rows=1200,
        row_type_counts='{"domain_transport":1200,"hard_anchor":4923}',
    )

    check = validate_t41_main_row(row)
    assert check["valid"], check
    assert row["method"] == PUBLIC_METHOD_ID
    assert row["method_id"] == PUBLIC_METHOD_ID
    assert row["public_method_name"] == PUBLIC_METHOD_NAME
    assert row["method_name"] == PUBLIC_METHOD_NAME
    assert row["ratio"] == 0.0025
    assert row["micro_f1"] == row["accuracy"]
    for field in [
        "storage_bytes",
        "domain_transport_active",
        "domain_transport_strength",
        "domain_transport_rows",
        "domain_row_frac",
        "domain_gap_before",
        "domain_gap_after",
        "domain_transport_gain",
        "domain_overfit_proxy",
        "score_v2_equivalent",
        "score_v3",
        "row_type_counts",
    ]:
        assert field in row

    legacy = dict(row)
    legacy["method"] = "products_uca_hybrid_mixup"
    assert not validate_t41_main_table([legacy])["valid"]


def test_t41_papers100m_one_cache_guard_for_promoted_rows() -> None:
    row = make_t41_row(
        dataset="ogbn-papers100M",
        requested_full_node_ratio=0.0005,
        condensed_nodes=55_530,
        num_classes=172,
        accuracy=0.59,
        macro_f1=0.30,
        valid_acc=0.59,
        selected_policy="auto_base",
        promotion_status="promoted",
        domain_gap_train_all=0.01,
        edge_cache_id="edge",
        sft_cache_id="sft",
        teacher_cache_id="teacher",
        reservoir_cache_id="reservoir",
        cache_reused=True,
        incremental_edge_scans_after_cache_build=0,
        teacher_cache_mode="topk8_tail",
        uses_dense_all_node_teacher_cache=False,
    )

    assert validate_t41_main_table([row])["valid"]

    bad = dict(row)
    bad["incremental_edge_scans_after_cache_build"] = 1
    assert not validate_t41_main_table([bad])["valid"]
