from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.run_t31_arxiv_semantic_sft import build_semantic_rows
from shadow_hgc.sft.semantic_sft_blocks import validate_semantic_cache_alignment


def test_t31_semantic_missing_raw_text_or_cache_blocks(tmp_path: Path) -> None:
    rows = build_semantic_rows(
        Namespace(
            seed=42,
            lm_models=["scibert"],
            raw_text_map="",
            node_id_to_paper_id="",
            use_precomputed_semantic_features="",
            semantic_cache_dir=str(tmp_path),
            build_semantic_sft=True,
            teacher_heads=["mlp"],
            enable_cns=True,
            hidden_dims=[8],
            epochs=1,
            run_long=False,
            device="cpu",
            semantic_device="cpu",
        )
    )
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "raw_text_or_semantic_cache_missing"
    assert rows[0]["uses_external_text_features"] is True


def test_t31_semantic_cache_alignment_logs_unmatched_nodes(tmp_path: Path) -> None:
    emb = tmp_path / "semantic.memmap"
    arr = np.memmap(emb, mode="w+", dtype=np.float32, shape=(3, 2))
    arr[:] = np.ones((3, 2), dtype=np.float32)
    arr.flush()
    diag = validate_semantic_cache_alignment(embedding_path=emb, shape=(3, 2), num_nodes=4, matched_nodes=3, min_match_rate=0.5)
    assert diag["semantic_unmatched_nodes"] == 1
    assert diag["semantic_match_rate"] == 0.75
    assert diag["blocked"] is False
