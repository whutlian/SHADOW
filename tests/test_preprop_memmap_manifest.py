import json

import numpy as np
import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop import PrepropBlockSpec, compute_preprop_blocks


def test_preprop_engine_writes_memmap_manifest_and_train_fit_stats(tmp_path):
    rel = DirectedRelation("paper", "cite_ref", "paper")
    manifest = compute_preprop_blocks(
        dataset_name="toy",
        target_type="paper",
        block_specs=[
            PrepropBlockSpec.self_block(name="X0", target_rows=[0, 1, 2], train_rows=[0, 1]),
            PrepropBlockSpec.typed_feature(name="X1_cite_ref", relation=rel, target_rows=[0, 1, 2], train_rows=[0, 1]),
        ],
        feature_provider={"paper": torch.arange(12, dtype=torch.float32).view(3, 4)},
        edge_store={rel: torch.tensor([[1, 2, 0], [0, 0, 2]], dtype=torch.long)},
        output_dir=str(tmp_path),
        dtype="float16",
        block_dim=4,
        edge_chunk_size=2,
        use_memmap=True,
        seed=42,
    )

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "toy"
    assert payload["target_type"] == "paper"
    assert payload["seed"] == 42
    assert payload["total_cache_bytes"] > 0
    assert payload["blocks"][0]["uses_logits"] is False
    assert payload["blocks"][1]["uses_dense_p2"] is False
    assert payload["blocks"][1]["stats_fit_scope"] == "train_target_rows"
    assert manifest.blocks[0].shape == [3, 4]

    block_meta = payload["blocks"][1]
    mmap = np.memmap(tmp_path / block_meta["path"], mode="r", dtype=np.float16, shape=tuple(block_meta["shape"]))
    assert mmap.shape == (3, 4)
    stats = json.loads((tmp_path / "block_X1_cite_ref_stats.json").read_text(encoding="utf-8"))
    assert stats["fit_rows"] == [0, 1]
    assert stats["fit_scope"] == "train_target_rows"
