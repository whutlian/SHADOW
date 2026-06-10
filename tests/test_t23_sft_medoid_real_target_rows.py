import torch

from shadow_hgc.condense.sft_condense import condense_sft_blocks


def test_t23_sft_medoid_selects_real_target_rows():
    blocks = {"self": torch.arange(24, dtype=torch.float32).view(6, 4)}
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    train_rows = torch.arange(6)
    result = condense_sft_blocks(
        blocks=blocks,
        signatures=blocks["self"],
        labels=labels,
        train_rows=train_rows,
        ratio=0.5,
        method="medoid",
    )
    assert result.selected_rows.numel() == result.condensed_labels.numel()
    assert set(result.selected_rows.tolist()) <= set(train_rows.tolist())
    assert torch.equal(result.condensed_blocks["self"], blocks["self"][result.selected_rows])
    assert result.diagnostics["real_target_rows"] is True
