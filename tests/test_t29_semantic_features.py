from __future__ import annotations

from pathlib import Path

import torch

from scripts.run_t29_arxiv_semantic_teacher import build_semantic_rows
from shadow_hgc.sft.semantic_arxiv_features import (
    load_arxiv_raw_text_map,
    read_semantic_cache_manifest,
    semantic_flags,
    write_semantic_cache_manifest,
)


def test_t29_semantic_loader_missing_text_blocks_cleanly(tmp_path: Path):
    result = load_arxiv_raw_text_map(search_paths=[tmp_path / "missing.jsonl"], precomputed_embedding_path=None)
    assert result.available is False
    assert result.failure_reason == "raw_text_missing"
    assert "provide" in result.actionable_message.lower()


def test_t29_semantic_cache_manifest_roundtrip(tmp_path: Path):
    manifest = write_semantic_cache_manifest(
        tmp_path,
        model_name="test-lm",
        embedding_path="embeddings.memmap",
        num_nodes=5,
        feature_dim=3,
        dtype="float32",
        cache_bytes=60,
    )
    loaded = read_semantic_cache_manifest(manifest)
    assert loaded["model_name"] == "test-lm"
    assert loaded["shape"] == [5, 3]
    assert loaded["cache_bytes"] == 60


def test_t29_semantic_flags_logged():
    flags = semantic_flags(model_name="specter", feature_dim=768, cache_bytes=1234, raw_text_encoded=True, encode_time=1.5)
    assert flags["uses_external_text_features"] is True
    assert flags["uses_raw_text"] is True
    assert flags["uses_lm_encoder"] is True
    assert flags["semantic_lm_model"] == "specter"
    assert flags["semantic_feature_dim"] == 768


def test_t29_semantic_no_fabricated_features(tmp_path: Path):
    args = type(
        "Args",
        (),
        {
            "seed": 42,
            "lm_models": ["specter"],
            "semantic_cache_dir": str(tmp_path),
            "raw_text_path": str(tmp_path / "missing.jsonl"),
            "use_precomputed_semantic_features": "",
            "smoke": False,
        },
    )()
    rows = build_semantic_rows(args)
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "raw_text_missing"
    assert rows[0]["accuracy"] == ""
