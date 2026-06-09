import torch

from shadow_hgc.training.safe_block_selection import train_with_validation_gated_blocks


def test_validation_gated_block_selection_drops_noise_and_keeps_useful_block():
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1])
    self_block = torch.full((8, 2), 0.5)
    useful = torch.nn.functional.one_hot(labels, num_classes=2).to(torch.float32)
    noise = torch.tensor(
        [
            [0.11, -0.2],
            [-0.3, 0.4],
            [0.5, 0.2],
            [0.1, 0.3],
            [-0.4, 0.2],
            [0.3, -0.5],
            [0.2, 0.2],
            [-0.1, 0.1],
        ]
    )

    result = train_with_validation_gated_blocks(
        {"self": self_block, "useful": useful, "noise": noise},
        labels,
        train_rows=torch.tensor([0, 1, 3, 4]),
        val_rows=torch.tensor([2, 5]),
        test_rows=torch.tensor([6, 7]),
        num_classes=2,
        seed=42,
        epochs=80,
        epsilon_acc=0.001,
        epsilon_f1=0.002,
    )

    decisions = {row["block_name"]: row["kept_or_dropped"] for row in result.block_diagnostics}
    assert decisions["useful"] == "kept"
    assert decisions["noise"] == "dropped"
    assert result.summary["final_val_acc"] >= result.summary["self_val_acc"]
