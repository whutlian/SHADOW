from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any


FORBIDDEN_PROMOTION_FLAGS: tuple[str, ...] = (
    "uses_diffusion",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_source_anchors",
    "uses_coverage_medoid",
    "uses_old_kd",
)


@dataclass(frozen=True)
class LogitCacheMeta:
    dataset: str
    variant: str
    seed: int
    num_target_nodes: int
    num_classes: int
    target_type: str
    split_hash: str | None
    feature_hash: str | None
    uses_diffusion: bool
    uses_dense_p2: bool
    uses_bounded_edges: bool
    uses_source_anchors: bool
    uses_coverage_medoid: bool
    uses_old_kd: bool
    accuracy: float | None
    macro_f1: float | None
    predicted_class_count: int | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LogitCacheMeta":
        names = {field.name for field in fields(cls)}
        values = {name: payload.get(name) for name in names}
        return cls(**values)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def forbidden_reasons(meta: LogitCacheMeta | dict[str, Any]) -> list[str]:
    payload = meta.to_dict() if isinstance(meta, LogitCacheMeta) else dict(meta)
    return [flag for flag in FORBIDDEN_PROMOTION_FLAGS if bool(payload.get(flag, False))]


def is_promotable_cache(meta: LogitCacheMeta | dict[str, Any]) -> bool:
    return len(forbidden_reasons(meta)) == 0
