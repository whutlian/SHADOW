from __future__ import annotations

from pathlib import Path

import numpy as np

from shadow_hgc.sft.arxiv_semantic_stt import validate_precomputed_semantic_memmap


def test_t34_precomputed_semantic_memmap_validation_enforces_shape_and_checksum(tmp_path: Path) -> None:
    emb = tmp_path / "semantic.memmap"
    arr = np.memmap(emb, mode="w+", dtype=np.float16, shape=(3, 2))
    arr[:] = np.ones((3, 2), dtype=np.float16)
    arr.flush()
    checksum = tmp_path / "node_order.sha256"
    checksum.write_text("abc123\n", encoding="utf-8")
    diag = validate_precomputed_semantic_memmap(
        memmap_path=emb,
        semantic_node_order_checksum=checksum,
        expected_node_order_checksum="abc123",
        num_nodes=3,
        semantic_dim=2,
        semantic_dtype="fp16",
    )
    assert diag["blocked"] is False
    assert diag["semantic_cache_memmap"] is True
    assert diag["semantic_features_are_frozen"] is True
    assert diag["lm_finetuned"] is False


def test_t34_missing_semantic_cache_blocks_without_fabrication(tmp_path: Path) -> None:
    diag = validate_precomputed_semantic_memmap(
        memmap_path=tmp_path / "missing.memmap",
        semantic_node_order_checksum="abc123",
        expected_node_order_checksum="abc123",
        num_nodes=3,
        semantic_dim=2,
        semantic_dtype="fp16",
    )
    assert diag["blocked"] is True
    assert diag["failure_reason"] == "raw_text_or_semantic_cache_missing"
