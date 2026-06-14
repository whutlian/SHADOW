from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RatioHardAnchorPolicy:
    ratio: float
    hard_anchor_frac: float
    rare_repair_frac: float
    boundary_frac: float
    lambda_hard: float
    soft_temperature: float
    lambda_prior: float = 0.02

    @property
    def core_frac(self) -> float:
        return max(0.0, 1.0 - self.rare_repair_frac - self.boundary_frac)

    def as_row_fields(self) -> dict[str, float]:
        return {
            "hard_anchor_frac": self.hard_anchor_frac,
            "rare_repair_frac": self.rare_repair_frac,
            "boundary_frac": self.boundary_frac,
            "lambda_hard": self.lambda_hard,
            "soft_temperature": self.soft_temperature,
            "lambda_prior": self.lambda_prior,
        }


def ratio_policy_v2(ratio: float) -> RatioHardAnchorPolicy:
    ratio = float(ratio)
    if ratio <= 0.00020 + 1e-12:
        return RatioHardAnchorPolicy(
            ratio=ratio,
            hard_anchor_frac=0.80,
            rare_repair_frac=0.12,
            boundary_frac=0.08,
            lambda_hard=0.75,
            soft_temperature=1.5,
        )
    if ratio <= 0.00050 + 1e-12:
        return RatioHardAnchorPolicy(
            ratio=ratio,
            hard_anchor_frac=0.65,
            rare_repair_frac=0.18,
            boundary_frac=0.17,
            lambda_hard=0.50,
            soft_temperature=1.5,
        )
    if ratio <= 0.00200 + 1e-12:
        return RatioHardAnchorPolicy(
            ratio=ratio,
            hard_anchor_frac=0.55,
            rare_repair_frac=0.18,
            boundary_frac=0.27,
            lambda_hard=0.35,
            soft_temperature=2.0,
        )
    return RatioHardAnchorPolicy(
        ratio=ratio,
        hard_anchor_frac=0.45,
        rare_repair_frac=0.15,
        boundary_frac=0.30,
        lambda_hard=0.25,
        soft_temperature=2.0,
    )


def boundary_weight_for_ratio(ratio: float) -> float:
    ratio = float(ratio)
    if ratio <= 0.00020 + 1e-12:
        return 0.08
    if ratio <= 0.00050 + 1e-12:
        return 0.18
    if ratio <= 0.00200 + 1e-12:
        return 0.30
    return 0.25
