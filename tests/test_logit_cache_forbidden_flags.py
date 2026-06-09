from shadow_hgc.logits.metadata import LogitCacheMeta, forbidden_reasons, is_promotable_cache


def _meta(**kwargs):
    fields = dict(
        dataset="toy",
        variant="row",
        seed=42,
        num_target_nodes=3,
        num_classes=2,
        target_type="paper",
        split_hash=None,
        feature_hash=None,
        uses_diffusion=False,
        uses_dense_p2=False,
        uses_bounded_edges=False,
        uses_source_anchors=False,
        uses_coverage_medoid=False,
        uses_old_kd=False,
        accuracy=None,
        macro_f1=None,
        predicted_class_count=None,
        created_at="now",
    )
    fields.update(kwargs)
    return LogitCacheMeta(**fields)


def test_forbidden_flags_block_promotion():
    meta = _meta(uses_diffusion=True, uses_bounded_edges=True)

    assert is_promotable_cache(meta) is False
    assert forbidden_reasons(meta) == ["uses_diffusion", "uses_bounded_edges"]


def test_safe_cache_is_promotable():
    assert is_promotable_cache(_meta()) is True
