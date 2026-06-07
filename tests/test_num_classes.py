import torch

from shadow_hgc.pipeline.core import infer_class_metadata


def test_num_classes_uses_all_nonnegative_labels_not_only_train_labels():
    labels = torch.tensor([0, 0, 1, 3, -1])
    train_idx = torch.tensor([0, 1, 2])
    test_idx = torch.tensor([3])

    metadata = infer_class_metadata(labels, train_idx, test_idx)

    assert metadata["num_classes_global"] == 4
    assert metadata["num_classes_train"] == 2
    assert metadata["train_label_classes"] == [0, 1]
    assert metadata["test_label_classes"] == [3]
