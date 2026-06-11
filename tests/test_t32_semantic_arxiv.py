from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.run_t32_arxiv_semantic_sft import build_semantic_rows
from shadow_hgc.sft.arxiv_semantic_cache_v2 import validate_semantic_memmap


def test_t32_semantic_missing_raw_text_or_cache_blocks(tmp_path: Path) -> None:
    rows = build_semantic_rows(
        Namespace(
            seed=42,
            lm_models=["scibert"],
            raw_text_map="",
            node_id_to_paper_id="",
            use_precomputed_semantic_features="",
            semantic_cache_dir=str(tmp_path),
            build_semantic_cache_if_missing=True,
            build_semantic_sft=True,
            teacher_heads=["semantic_mlp"],
            enable_cns=True,
            hidden_dims=[8],
            epochs=1,
            run_long=False,
            device="cpu",
            semantic_device="cpu",
            temporal_decay_gammas=[0.01],
        )
    )
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "raw_text_or_semantic_cache_missing"
    assert rows[0]["uses_external_text_features"] is True


def test_t32_semantic_memmap_shape_validation(tmp_path: Path) -> None:
    emb = tmp_path / "semantic.memmap"
    arr = np.memmap(emb, mode="w+", dtype=np.float32, shape=(3, 2))
    arr[:] = np.ones((3, 2), dtype=np.float32)
    arr.flush()
    diag = validate_semantic_memmap(embedding_path=emb, shape=(3, 2), num_nodes=3, dim=2)
    assert diag["blocked"] is False
    assert diag["semantic_dim"] == 2
    assert diag["semantic_cache_bytes"] > 0
