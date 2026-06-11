from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.run_t30_arxiv_semantic_teacher import build_semantic_rows
from shadow_hgc.sft.semantic_arxiv_features import (
    load_arxiv_raw_text_map,
    read_semantic_cache_manifest,
    write_semantic_cache_manifest,
)


def test_t30_semantic_missing_raw_text_blocks_with_action(tmp_path: Path) -> None:
    rows = build_semantic_rows(
        Namespace(
            seed=42,
            lm_models=["scibert"],
            raw_text_map=tmp_path / "missing.jsonl",
            semantic_cache_dir=tmp_path / "semantic",
            use_precomputed_semantic_features="",
            build_semantic_sft=True,
            teacher_heads=["mlp"],
        )
    )
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "raw_text_missing"
    assert "--raw-text-map" in rows[0]["notes"]


def test_t30_semantic_manifest_records_node_ordering(tmp_path: Path) -> None:
    emb = tmp_path / "semantic.memmap"
    np.memmap(emb, mode="w+", dtype=np.float16, shape=(3, 4))[:] = 0.0
    manifest = write_semantic_cache_manifest(
        tmp_path,
        model_name="scibert",
        embedding_path=str(emb),
        num_nodes=3,
        feature_dim=4,
        dtype="float16",
        cache_bytes=24,
        node_ordering="ogb_node_id",
        text_fields=["title", "abstract"],
    )
    loaded = read_semantic_cache_manifest(manifest)
    assert loaded["node_ordering"] == "ogb_node_id"
    assert loaded["text_fields"] == ["title", "abstract"]


def test_t30_raw_text_loader_does_not_fabricate_missing_text(tmp_path: Path) -> None:
    result = load_arxiv_raw_text_map(search_paths=[tmp_path / "absent.csv"])
    assert result.available is False
    assert result.failure_reason == "raw_text_missing"
    assert result.text_map == {}
