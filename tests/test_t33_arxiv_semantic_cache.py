from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.run_t33_arxiv_semantic_sft import build_semantic_rows
from shadow_hgc.sft.arxiv_semantic_cache_v3 import write_semantic_cache_metadata, validate_semantic_cache_v3


def test_t33_semantic_missing_inputs_block_clearly(tmp_path: Path) -> None:
    rows = build_semantic_rows(
        Namespace(
            seed=42,
            lm_models=["scibert"],
            raw_text_map="",
            node_id_to_paper_id="",
            use_precomputed_semantic_features="",
            semantic_dim=0,
            semantic_cache_dir=str(tmp_path),
            build_semantic_sft=True,
            enable_cns=True,
            hidden_dims=[8],
            epochs=1,
            device="cpu",
            semantic_device="cpu",
            run_long=False,
        )
    )
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "raw_text_or_semantic_cache_missing"
    assert rows[0]["uses_external_text_features"] is True


def test_t33_semantic_cache_metadata_validation(tmp_path: Path) -> None:
    emb = tmp_path / "semantic.fp16.memmap"
    arr = np.memmap(emb, mode="w+", dtype=np.float16, shape=(3, 2))
    arr[:] = np.ones((3, 2), dtype=np.float16)
    arr.flush()
    meta = write_semantic_cache_metadata(tmp_path / "semantic.json", embedding_path=emb, model_name="e5", shape=(3, 2))
    diag = validate_semantic_cache_v3(metadata_path=meta, expected_num_nodes=3)
    assert diag["blocked"] is False
    assert diag["semantic_encoder"] == "e5"
    assert diag["semantic_dim"] == 2
    assert diag["semantic_cache_bytes"] > 0
