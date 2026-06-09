from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class BlockStandardizer:
    block_name: str
    mean: torch.Tensor
    std: torch.Tensor
    fit_rows: list[int]
    fit_scope: str = "train_target_rows"
    frozen: bool = False
    eps: float = 1e-6

    @classmethod
    def fit(
        cls,
        block: torch.Tensor,
        *,
        train_rows: torch.Tensor | list[int],
        block_name: str,
        eps: float = 1e-6,
    ) -> "BlockStandardizer":
        if block.ndim != 2:
            raise ValueError("block must have shape [num_rows, dim]")
        rows = torch.as_tensor(train_rows, dtype=torch.long, device=block.device)
        if rows.numel() == 0:
            raise ValueError("train_rows must select at least one row")
        scoped = block[rows].detach().to(torch.float32)
        return cls(
            block_name=str(block_name),
            mean=scoped.mean(dim=0).cpu(),
            std=scoped.std(dim=0, unbiased=False).clamp_min(float(eps)).cpu(),
            fit_rows=[int(value) for value in rows.detach().cpu().tolist()],
            fit_scope="train_target_rows",
            frozen=False,
            eps=float(eps),
        )

    def freeze(self) -> "BlockStandardizer":
        return replace(self, frozen=True)

    def refit(self, block: torch.Tensor, *, train_rows: torch.Tensor | list[int]) -> "BlockStandardizer":
        if self.frozen:
            raise RuntimeError("block stats are frozen and cannot be refit")
        return self.fit(block, train_rows=train_rows, block_name=self.block_name, eps=self.eps)

    def transform(self, block: torch.Tensor) -> torch.Tensor:
        if block.ndim != 2:
            raise ValueError("block must have shape [num_rows, dim]")
        if int(block.shape[1]) != int(self.mean.numel()):
            raise ValueError(f"{self.block_name}: expected dim {self.mean.numel()}, got {block.shape[1]}")
        mean = self.mean.to(device=block.device, dtype=block.dtype)
        std = self.std.to(device=block.device, dtype=block.dtype).clamp_min(float(self.eps))
        return torch.nan_to_num((block - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    def to_json(self) -> dict:
        fit_rows_payload: dict[str, object]
        if len(self.fit_rows) <= 1024:
            fit_rows_payload = {"fit_rows": self.fit_rows}
        else:
            fit_rows_payload = {
                "fit_rows": [],
                "fit_rows_count": len(self.fit_rows),
                "fit_rows_head": self.fit_rows[:16],
                "fit_rows_tail": self.fit_rows[-16:],
            }
        return {
            "block_name": self.block_name,
            "fit_scope": self.fit_scope,
            "frozen": bool(self.frozen),
            "eps": float(self.eps),
            "mean": [float(value) for value in self.mean.detach().cpu().tolist()],
            "std": [float(value) for value in self.std.detach().cpu().tolist()],
            "norm_mean": float(torch.linalg.vector_norm(self.mean.to(torch.float32)).item()),
            "std_mean": float(self.std.to(torch.float32).mean().item()) if self.std.numel() else 0.0,
            **fit_rows_payload,
        }
