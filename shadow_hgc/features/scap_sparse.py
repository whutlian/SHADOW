from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SCAPTopK:
    class_ids: torch.Tensor
    values: torch.Tensor
    num_rows: int
    num_classes: int
    metadata: dict


def scap_topk_from_dense(dense: torch.Tensor, *, top_k: int = 8) -> SCAPTopK:
    top_k = min(int(top_k), int(dense.shape[1]))
    values, class_ids = torch.topk(dense.to(torch.float32).abs(), k=top_k, dim=1)
    signed_values = torch.gather(dense.to(torch.float32), 1, class_ids)
    return SCAPTopK(
        class_ids=class_ids.to(torch.int64),
        values=signed_values,
        num_rows=int(dense.shape[0]),
        num_classes=int(dense.shape[1]),
        metadata={"dense_or_sparse": "sparse_topk", "top_k": int(top_k), "uses_train_labels_only": True},
    )


def dense_from_scap_topk(sparse: SCAPTopK, *, num_classes: int | None = None) -> torch.Tensor:
    classes = int(num_classes if num_classes is not None else sparse.num_classes)
    dense = torch.zeros(int(sparse.num_rows), classes, dtype=torch.float32, device=sparse.values.device)
    rows = torch.arange(int(sparse.num_rows), device=sparse.values.device).unsqueeze(1).expand_as(sparse.class_ids)
    dense[rows, sparse.class_ids.to(torch.long)] = sparse.values.to(torch.float32)
    return dense
