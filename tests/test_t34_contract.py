from __future__ import annotations

from shadow_hgc.sft.t34_contract import T34_REQUIRED_FIELDS, apply_t34_promotion_guard, make_t34_row, validate_t34_row


def test_t34_schema_contains_stt_flags() -> None:
    required = set(T34_REQUIRED_FIELDS)
    for field in [
        "uses_teacher_probs_as_input",
        "soft_target_only",
        "uses_dense_nxc_teacher_cache",
        "lambda_cover",
        "lambda_calib",
        "semantic_features_are_frozen",
        "lm_finetuned",
    ]:
        assert field in required


def test_t34_sota_row_rejects_teacher_probs_as_input() -> None:
    row = make_t34_row(
        dataset="Reddit",
        method="reddit_stt_gamlp_ratio_v2",
        seed=42,
        requested_full_node_ratio=0.005,
        accuracy=0.94,
        macro_f1=0.91,
        status="completed_long",
        promotion_track="sota_chase",
        promotion_status="promoted",
        uses_teacher_probs=True,
        uses_teacher_logits=True,
        uses_logits_as_input=False,
        uses_teacher_probs_as_input=True,
        soft_target_only=True,
        teacher_cache_mode="dense_fp16",
        teacher_cache_bytes=1024,
    )
    guarded = apply_t34_promotion_guard(row)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "uses_teacher_probs_as_input" in guarded["failure_reason"]


def test_t34_safe_main_rejects_teacher_and_semantic_features() -> None:
    row = make_t34_row(
        dataset="ogbn-products",
        method="products_stt_official",
        seed=42,
        requested_full_node_ratio=0.005,
        accuracy=0.8,
        macro_f1=0.42,
        status="completed_long",
        promotion_track="safe_main",
        promotion_status="promoted",
        uses_teacher_probs=True,
        uses_external_text_features=True,
    )
    result = validate_t34_row(row)
    assert not result["valid"]
    assert "uses_teacher_probs" in result["forbidden_flags"]
    assert "uses_external_text_features" in result["forbidden_flags"]


def test_t34_ultra_rejects_dense_nxc_cache_for_promoted_rows() -> None:
    row = make_t34_row(
        dataset="MAG240M",
        method="ultra_stt_planner_MAG240M",
        seed=0,
        status="completed_dry_run",
        promotion_track="ultra_planner",
        promotion_status="promoted",
        teacher_cache_mode="dense_fp16",
        uses_dense_nxc_teacher_cache=True,
    )
    guarded = apply_t34_promotion_guard(row)
    assert guarded["promotion_status"] == "blocked_forbidden"
    assert "ultra_dense_nxc_teacher_cache" in guarded["failure_reason"]
