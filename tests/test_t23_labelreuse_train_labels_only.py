import torch

from shadow_hgc.preprop.label_reuse_v2 import compute_label_reuse_v2_blocks


def test_t23_labelreuse_v2_uses_train_labels_only():
    edge = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    blocks, diagnostics = compute_label_reuse_v2_blocks(
        relation_blocks={"cite_ref": edge},
        labels=torch.tensor([0, 1, 1, 0]),
        train_target_ids=torch.tensor([0]),
        num_target_nodes=4,
        num_classes=2,
        edge_chunk_size=2,
    )
    y0 = blocks["Y0_train_masked"]
    assert torch.equal(y0[0], torch.tensor([1.0, 0.0]))
    assert torch.equal(y0[1:], torch.zeros(3, 2))
    assert diagnostics["uses_valid_labels"] is False
    assert diagnostics["uses_test_labels"] is False
    assert "Y4_mix" in blocks
    assert "Yres1_mix" in blocks
