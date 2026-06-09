import pytest
import torch

from shadow_hgc.features.block_stats import BlockStandardizer


def test_block_stats_fit_on_train_rows_and_freeze_for_later_rows():
    block = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [100.0, 200.0],
        ]
    )
    stats = BlockStandardizer.fit(block, train_rows=torch.tensor([0, 1]), block_name="X0")
    assert stats.fit_scope == "train_target_rows"
    frozen = stats.freeze()

    transformed = frozen.transform(block)
    torch.testing.assert_close(transformed[:2].mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0.0)

    with pytest.raises(RuntimeError, match="frozen"):
        frozen.refit(block, train_rows=torch.tensor([2]))
