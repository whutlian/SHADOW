from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3
from shadow_hgc.sft.stc import BlockSpec, apply_tanh_bounded_delta, delta_bound_ratios
from shadow_hgc.sft.stc_losses import weighted_cross_entropy
from shadow_hgc.train.lazy_sft_memmap import _load_block_stats_into_model, evaluate_lazy_sft
from shadow_hgc.train.train_sft_teacher import sft_loss


@dataclass
class STCOptimizationResult:
    z_syn: torch.Tensor
    y_syn: torch.Tensor
    initial_real_loss: float
    final_real_loss: float
    delta_bound_ratios: dict[str, float]
    used_valid_labels: bool = False
    used_test_labels: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class STCFinalTrainResult:
    metrics: dict[str, Any]
    valid_metrics: dict[str, Any]
    training_time_s: float
    inference_time_s: float
    peak_cpu_ram_gb: float
    peak_gpu_ram_gb: float


class FlatTableHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, head_type: str = "hidden_mlp") -> None:
        super().__init__()
        if head_type not in {"hidden_mlp", "linear", "residual_gated"}:
            raise ValueError(f"unsupported head_type: {head_type}")
        self.head_type = head_type
        if head_type == "linear":
            self.net = nn.Linear(input_dim, num_classes)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _as_tensor(value: torch.Tensor | np.ndarray, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def split_flat_table_by_blocks(z_flat: torch.Tensor, block_dims: dict[str, int]) -> dict[str, torch.Tensor]:
    blocks: dict[str, torch.Tensor] = {}
    offset = 0
    for name, dim in block_dims.items():
        end = offset + int(dim)
        blocks[name] = z_flat[:, offset:end]
        offset = end
    if offset != int(z_flat.shape[1]):
        raise ValueError(f"flat table dim {z_flat.shape[1]} does not match block dims total {offset}")
    return blocks


def _prototype_logits(z_real: torch.Tensor, z_syn: torch.Tensor, y_syn: torch.Tensor, num_classes: int) -> torch.Tensor:
    z_real_n = F.normalize(z_real, dim=1)
    z_syn_n = F.normalize(z_syn, dim=1)
    sims = z_real_n @ z_syn_n.t()
    logits: list[torch.Tensor] = []
    for cls in range(int(num_classes)):
        mask = y_syn == cls
        if mask.any():
            logits.append(torch.logsumexp(8.0 * sims[:, mask], dim=1) - torch.log(mask.sum().float()))
        else:
            logits.append(torch.full((z_real.shape[0],), -30.0, device=z_real.device, dtype=z_real.dtype))
    return torch.stack(logits, dim=1)


def _sample_batch(z_real: torch.Tensor, y_real: torch.Tensor, batch_size: int | None, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= z_real.shape[0]:
        return z_real, y_real
    idx = torch.randperm(z_real.shape[0], generator=generator, device=z_real.device)[: int(batch_size)]
    return z_real[idx], y_real[idx]


def optimize_trainable_delta(
    z_init: torch.Tensor | np.ndarray,
    y_syn: torch.Tensor | np.ndarray,
    z_real: torch.Tensor | np.ndarray,
    y_real: torch.Tensor | np.ndarray,
    blocks: list[BlockSpec] | tuple[BlockSpec, ...],
    *,
    num_classes: int,
    rho: float,
    outer_steps: int,
    lr: float,
    seed: int,
    real_batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> STCOptimizationResult:
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    z_init_t = _as_tensor(z_init, dtype=torch.float32, device=dev)
    y_syn_t = _as_tensor(y_syn, dtype=torch.long, device=dev)
    z_real_t = _as_tensor(z_real, dtype=torch.float32, device=dev)
    y_real_t = _as_tensor(y_real, dtype=torch.long, device=dev)
    raw_delta = torch.zeros_like(z_init_t, requires_grad=True)
    optimizer = torch.optim.Adam([raw_delta], lr=float(lr))
    generator = torch.Generator(device=dev).manual_seed(int(seed) + 17)
    with torch.no_grad():
        init_z, init_y = _sample_batch(z_real_t, y_real_t, real_batch_size, generator)
        initial_logits = _prototype_logits(init_z, z_init_t, y_syn_t, int(num_classes))
        initial_loss = float(F.cross_entropy(initial_logits, init_y).item())
    for _ in range(int(outer_steps)):
        batch_z, batch_y = _sample_batch(z_real_t, y_real_t, real_batch_size, generator)
        z_syn, _ = apply_tanh_bounded_delta(z_init_t, raw_delta, blocks, rho=float(rho))
        logits = _prototype_logits(batch_z, z_syn, y_syn_t, int(num_classes))
        loss = F.cross_entropy(logits, batch_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        z_syn, delta = apply_tanh_bounded_delta(z_init_t, raw_delta, blocks, rho=float(rho))
        final_z, final_y = _sample_batch(z_real_t, y_real_t, real_batch_size, generator)
        final_logits = _prototype_logits(final_z, z_syn, y_syn_t, int(num_classes))
        final_loss = float(F.cross_entropy(final_logits, final_y).item())
        ratios = delta_bound_ratios(z_init_t, delta, blocks)
    return STCOptimizationResult(z_syn.detach().cpu(), y_syn_t.detach().cpu(), initial_loss, final_loss, ratios)


def optimize_gradient_matching(
    z_init: torch.Tensor | np.ndarray,
    y_syn: torch.Tensor | np.ndarray,
    z_real: torch.Tensor | np.ndarray,
    y_real: torch.Tensor | np.ndarray,
    blocks: list[BlockSpec] | tuple[BlockSpec, ...],
    *,
    num_classes: int,
    rho: float,
    outer_steps: int,
    real_batch_size: int,
    lr: float,
    seed: int,
    gm_num_heads: int = 1,
    hidden_dim: int = 32,
    device: str | torch.device = "cpu",
) -> STCOptimizationResult:
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    z_init_t = _as_tensor(z_init, dtype=torch.float32, device=dev)
    y_syn_t = _as_tensor(y_syn, dtype=torch.long, device=dev)
    z_real_t = _as_tensor(z_real, dtype=torch.float32, device=dev)
    y_real_t = _as_tensor(y_real, dtype=torch.long, device=dev)
    raw_delta = torch.zeros_like(z_init_t, requires_grad=True)
    optimizer = torch.optim.Adam([raw_delta], lr=float(lr))
    generator = torch.Generator(device=dev).manual_seed(int(seed) + 31)
    with torch.no_grad():
        init_z, init_y = _sample_batch(z_real_t, y_real_t, int(real_batch_size), generator)
        initial_logits = _prototype_logits(init_z, z_init_t, y_syn_t, int(num_classes))
        initial_loss = float(F.cross_entropy(initial_logits, init_y).item())
    for step in range(int(outer_steps)):
        total_loss = z_init_t.new_tensor(0.0)
        for head_id in range(int(gm_num_heads)):
            torch.manual_seed(int(seed) * 1000 + step * 17 + head_id)
            head = FlatTableHead(z_init_t.shape[1], int(num_classes), hidden_dim=hidden_dim).to(dev)
            params = tuple(head.parameters())
            batch_z, batch_y = _sample_batch(z_real_t, y_real_t, int(real_batch_size), generator)
            z_syn, _ = apply_tanh_bounded_delta(z_init_t, raw_delta, blocks, rho=float(rho))
            syn_loss = weighted_cross_entropy(head(z_syn), y_syn_t)
            real_loss = weighted_cross_entropy(head(batch_z), batch_y)
            syn_grads = torch.autograd.grad(syn_loss, params, create_graph=True)
            real_grads = torch.autograd.grad(real_loss, params, create_graph=False)
            for syn_grad, real_grad in zip(syn_grads, real_grads):
                total_loss = total_loss + F.mse_loss(syn_grad, real_grad.detach())
        optimizer.zero_grad(set_to_none=True)
        (total_loss / max(1, int(gm_num_heads))).backward()
        optimizer.step()
    with torch.no_grad():
        z_syn, delta = apply_tanh_bounded_delta(z_init_t, raw_delta, blocks, rho=float(rho))
        final_z, final_y = _sample_batch(z_real_t, y_real_t, int(real_batch_size), generator)
        final_logits = _prototype_logits(final_z, z_syn, y_syn_t, int(num_classes))
        final_loss = float(F.cross_entropy(final_logits, final_y).item())
        ratios = delta_bound_ratios(z_init_t, delta, blocks)
    return STCOptimizationResult(z_syn.detach().cpu(), y_syn_t.detach().cpu(), initial_loss, final_loss, ratios)


def optimize_outer_loop(*args, **kwargs) -> STCOptimizationResult:
    return optimize_trainable_delta(*args, **kwargs)


def _iter_index_batches(num_rows: int, *, batch_size: int, shuffle: bool, seed: int, device: torch.device) -> list[torch.Tensor]:
    order = torch.arange(int(num_rows), dtype=torch.long, device=device)
    if shuffle:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        order = order[torch.randperm(order.numel(), generator=generator, device=device)]
    return [order[start : start + int(batch_size)] for start in range(0, int(order.numel()), int(batch_size))]


def train_sft_teacher_on_synthetic_table(
    *,
    store,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    valid_rows: torch.Tensor,
    test_rows: torch.Tensor,
    z_syn: torch.Tensor | np.ndarray,
    y_syn: torch.Tensor | np.ndarray,
    num_classes: int,
    device: str | torch.device,
    model_type: str = "sagn_lite_v4",
    hidden_dim: int = 128,
    dropout: float = 0.3,
    epochs: int = 80,
    batch_size: int = 4096,
    eval_batch_size: int = 65536,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    loss_type: str = "sqrt_weighted_ce",
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    seed: int = 42,
) -> STCFinalTrainResult:
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    target_device = torch.device(device)
    z_syn_t = _as_tensor(z_syn, dtype=torch.float32, device=target_device)
    y_syn_t = _as_tensor(y_syn, dtype=torch.long, device=target_device)
    labels_cpu = labels.to(torch.long).cpu()
    train_labels = labels_cpu[train_rows.to(torch.long).cpu()].to(target_device)
    model = SFTTeacherV3(
        store.block_dims,
        num_classes=int(num_classes),
        model_type=model_type,
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(target_device)
    _load_block_stats_into_model(model, store)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.default_rng(int(seed))
    for epoch in range(int(epochs)):
        model.train()
        for idx in _iter_index_batches(z_syn_t.shape[0], batch_size=int(batch_size), shuffle=True, seed=int(seed) + epoch, device=target_device):
            optimizer.zero_grad(set_to_none=True)
            blocks = split_flat_table_by_blocks(z_syn_t[idx], store.block_dims)
            y = y_syn_t[idx]
            if float(mixup_alpha) > 0.0 and int(idx.numel()) > 1:
                lam = float(rng.beta(float(mixup_alpha), float(mixup_alpha)))
                perm = torch.randperm(int(idx.numel()), device=target_device)
                mixed_blocks = {name: lam * value + (1.0 - lam) * value[perm] for name, value in blocks.items()}
                logits = model(mixed_blocks)
                loss = lam * sft_loss(logits, y, loss_type=loss_type, train_labels=train_labels, label_smoothing=float(label_smoothing))
                loss = loss + (1.0 - lam) * sft_loss(logits, y[perm], loss_type=loss_type, train_labels=train_labels, label_smoothing=float(label_smoothing))
            else:
                logits = model(blocks)
                loss = sft_loss(logits, y, loss_type=loss_type, train_labels=train_labels, label_smoothing=float(label_smoothing))
            loss.backward()
            optimizer.step()
    train_s = float(time.perf_counter() - started)
    valid_metrics = evaluate_lazy_sft(
        model,
        store,
        labels_cpu,
        valid_rows.to(torch.long).cpu(),
        num_classes=int(num_classes),
        batch_size=int(eval_batch_size),
        device=target_device,
    )
    infer_started = time.perf_counter()
    metrics = evaluate_lazy_sft(
        model,
        store,
        labels_cpu,
        test_rows.to(torch.long).cpu(),
        num_classes=int(num_classes),
        batch_size=int(eval_batch_size),
        device=target_device,
    )
    infer_s = float(time.perf_counter() - infer_started)
    return STCFinalTrainResult(
        metrics=metrics,
        valid_metrics=valid_metrics,
        training_time_s=train_s,
        inference_time_s=infer_s,
        peak_cpu_ram_gb=current_cpu_ram_bytes() / (1024**3),
        peak_gpu_ram_gb=current_gpu_ram_bytes() / (1024**3),
    )


def train_flat_head(
    z_syn: torch.Tensor | np.ndarray,
    y_syn: torch.Tensor | np.ndarray,
    *,
    num_classes: int,
    weights: torch.Tensor | np.ndarray | None = None,
    hidden_dim: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> FlatTableHead:
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    z_syn_t = _as_tensor(z_syn, dtype=torch.float32, device=dev)
    y_syn_t = _as_tensor(y_syn, dtype=torch.long, device=dev)
    weights_t = None if weights is None else _as_tensor(weights, dtype=torch.float32, device=dev)
    model = FlatTableHead(z_syn_t.shape[1], int(num_classes), hidden_dim=int(hidden_dim)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    for _ in range(int(epochs)):
        loss = weighted_cross_entropy(model(z_syn_t), y_syn_t, weights_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def predict_flat_head(model: nn.Module, z: torch.Tensor | np.ndarray, *, batch_size: int = 8192, device: str | torch.device = "cpu") -> np.ndarray:
    dev = torch.device(device)
    z_t = _as_tensor(z, dtype=torch.float32, device=dev)
    model = model.to(dev)
    preds: list[torch.Tensor] = []
    for start in range(0, z_t.shape[0], int(batch_size)):
        logits = model(z_t[start : start + int(batch_size)])
        preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds, dim=0).numpy()
