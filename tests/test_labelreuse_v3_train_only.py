import torch

from shadow_hgc.preprop.label_reuse_v3 import compute_label_reuse_v3_blocks


def test_t24_labelreuse_v3_train_only_policy():
    blocks, diag = compute_label_reuse_v3_blocks(
        relation_blocks={"cite_ref": torch.tensor([[0, 1], [1, 2]], dtype=torch.long)},
        labels=torch.tensor([0, 1, 1]),
        train_target_ids=torch.tensor([0]),
        num_target_nodes=3,
        num_classes=2,
        edge_chunk_size=1,
    )
    assert blocks["Y0_train_masked"][0, 0] == 1
    assert blocks["Y0_train_masked"][1:].sum() == 0
    assert "Yres1" in blocks
    assert diag["label_reuse_version"] == "v3"
    assert diag["uses_valid_labels"] is False
    assert diag["uses_test_labels"] is False
