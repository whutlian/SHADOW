import torch

from shadow_hgc.training.safe_block_selection import train_with_validation_gated_blocks
from shadow_hgc.train.block_selection import select_t2_safe_blocks


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


def test_t2_safe_block_selection_logs_group_decisions_and_non_regression():
    rows = select_t2_safe_blocks(
        dataset="toy",
        baseline={"accuracy": 0.5, "macro_f1": 0.4},
        candidates=[
            {"block_group": "B1_typed", "valid_acc": 0.55, "test_acc": 0.6, "gate_value": 0.8, "block_dim": 4, "cache_bytes": 128},
            {"block_group": "B2_noise", "valid_acc": 0.49, "test_acc": 0.7, "gate_value": 0.1, "block_dim": 4, "cache_bytes": 128},
        ],
    )

    decisions = {row["block_group"]: row["kept_or_dropped"] for row in rows}
    assert decisions["B1_typed"] == "kept"
    assert decisions["B2_noise"] == "dropped"
    assert rows[0]["branch_test_acc_debug"] == 0.6
    assert rows[0]["uses_logits_as_input"] is False
