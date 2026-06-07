from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.data.schemas import DirectedRelation


@contextmanager
def trusted_ogb_torch_load():
    """Allow trusted OGB/PyG processed files to load under PyTorch 2.6+."""

    original_load = torch.load

    def load_trusted_ogb_file(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_trusted_ogb_file
    try:
        yield
    finally:
        torch.load = original_load


def build_homogeneous_special_case(
    *,
    dataset_name: str,
    target_type: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    forward_name: str,
    reverse_name: str,
) -> HeteroGraphData:
    forward = DirectedRelation(target_type, forward_name, target_type)
    reverse = DirectedRelation(target_type, reverse_name, target_type)
    labels = labels.squeeze().to(torch.long)
    return HeteroGraphData(
        dataset_name=dataset_name,
        target_type=target_type,
        node_features={target_type: x.to(torch.float32)},
        edge_index={
            forward: edge_index.to(torch.long),
            reverse: torch.stack([edge_index[1], edge_index[0]], dim=0).to(torch.long),
        },
        labels=labels,
        train_idx=train_idx.to(torch.long),
        val_idx=val_idx.to(torch.long),
        test_idx=test_idx.to(torch.long),
        relations=[forward, reverse],
        num_nodes={target_type: int(x.shape[0])},
    )


def load_ogb_node_property_dataset(
    dataset_name: str,
    *,
    root: str | Path = "dataset",
    download: bool = False,
) -> HeteroGraphData:
    dataset_dir = Path(root) / dataset_name.replace("-", "_")
    ogb_dir = Path(root) / dataset_name
    if not download and not dataset_dir.exists() and not ogb_dir.exists():
        raise FileNotFoundError(
            f"{dataset_name} is not present under {root}; pass --download to run OGB download"
        )
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise RuntimeError("ogb is required for medium dataset loading") from exc

    with trusted_ogb_torch_load():
        dataset = PygNodePropPredDataset(name=dataset_name, root=str(root))
        data = dataset[0]
        split = dataset.get_idx_split()
    if dataset_name == "ogbn-arxiv":
        target_type = "paper"
        forward_name = "cite_ref"
        reverse_name = "cited_by"
    elif dataset_name == "ogbn-products":
        target_type = "product"
        forward_name = "co_purchase"
        reverse_name = "co_purchased_by"
    else:
        raise ValueError(f"unsupported OGB dataset: {dataset_name}")
    return build_homogeneous_special_case(
        dataset_name=dataset_name,
        target_type=target_type,
        x=data.x,
        edge_index=data.edge_index,
        labels=data.y,
        train_idx=split["train"],
        val_idx=split["valid"],
        test_idx=split["test"],
        forward_name=forward_name,
        reverse_name=reverse_name,
    )
