import numpy as np

from shadow_hgc.data.memmap import create_memmap_feature_store, source_id_block_gather


def test_memmap_block_gather_preserves_duplicate_source_ids(tmp_path):
    path = tmp_path / "features.npy"
    data = np.arange(24, dtype=np.float32).reshape(12, 2)
    store = create_memmap_feature_store(path, data)
    source_ids = np.array([5, 1, 5, 2, 1, 6], dtype=np.int64)

    gathered, stats = source_id_block_gather(store, source_ids, block_size=2)

    assert np.array_equal(gathered, data[source_ids])
    assert stats["num_blocks"] >= 1
