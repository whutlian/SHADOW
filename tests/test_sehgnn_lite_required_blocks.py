from __future__ import annotations

from shadow_hgc.audit.schema_checks import require_nonempty_feature_blocks


def test_sehgnn_lite_required_blocks_reject_empty_block_lists():
    checks = require_nonempty_feature_blocks(
        {
            "model_type": "sehgnn_lite",
            "metapath_blocks": [],
            "block_dims": {},
            "feature_blocks": ["self"],
        },
        required_prefix="metapath",
    )

    assert checks["valid"] is False
    assert "metapath_blocks_empty" in checks["reasons"]


def test_sehgnn_lite_required_blocks_accept_positive_dims():
    checks = require_nonempty_feature_blocks(
        {
            "model_type": "sehgnn_lite",
            "metapath_blocks": ["APA"],
            "block_dims": {"APA": 8},
            "feature_blocks": ["self", "APA"],
        },
        required_prefix="metapath",
    )

    assert checks["valid"] is True
