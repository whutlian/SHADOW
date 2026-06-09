from __future__ import annotations

import torch


def temperature_scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    return logits.to(torch.float32) / float(temperature)


def probabilities_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.softmax(temperature_scale_logits(logits, temperature), dim=1)
