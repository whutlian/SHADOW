from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from shadow_hgc.data.schemas import DirectedRelation


def _rows(value: list[int] | tuple[int, ...] | None) -> list[int] | None:
    if value is None:
        return None
    return [int(row) for row in value]


def _relation_payload(relation: DirectedRelation | None) -> dict[str, str] | None:
    if relation is None:
        return None
    return {
        "source_type": relation.source_type,
        "relation_name": relation.relation_name,
        "destination_type": relation.destination_type,
    }


@dataclass(frozen=True)
class PrepropBlockSpec:
    name: str
    kind: str
    relation: DirectedRelation | None = None
    path_schema: tuple[DirectedRelation, ...] = ()
    target_rows: list[int] | None = None
    train_rows: list[int] | None = None
    top_k: int = 8
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def self_block(
        cls,
        *,
        name: str = "X0",
        target_rows: list[int] | tuple[int, ...] | None = None,
        train_rows: list[int] | tuple[int, ...] | None = None,
    ) -> "PrepropBlockSpec":
        return cls(name=name, kind="self", target_rows=_rows(target_rows), train_rows=_rows(train_rows))

    @classmethod
    def typed_feature(
        cls,
        *,
        name: str,
        relation: DirectedRelation,
        target_rows: list[int] | tuple[int, ...] | None = None,
        train_rows: list[int] | tuple[int, ...] | None = None,
    ) -> "PrepropBlockSpec":
        return cls(name=name, kind="typed_feature", relation=relation, target_rows=_rows(target_rows), train_rows=_rows(train_rows))

    @classmethod
    def lad(
        cls,
        *,
        name: str,
        relation: DirectedRelation,
        target_rows: list[int] | tuple[int, ...] | None = None,
        train_rows: list[int] | tuple[int, ...] | None = None,
        top_k: int = 8,
    ) -> "PrepropBlockSpec":
        return cls(name=name, kind="lad", relation=relation, target_rows=_rows(target_rows), train_rows=_rows(train_rows), top_k=int(top_k))

    @classmethod
    def metapath_feature(
        cls,
        *,
        name: str,
        path_schema: list[DirectedRelation] | tuple[DirectedRelation, ...],
        target_rows: list[int] | tuple[int, ...] | None = None,
        train_rows: list[int] | tuple[int, ...] | None = None,
    ) -> "PrepropBlockSpec":
        return cls(
            name=name,
            kind="metapath_feature",
            path_schema=tuple(path_schema),
            target_rows=_rows(target_rows),
            train_rows=_rows(train_rows),
        )

    @classmethod
    def structure(
        cls,
        *,
        name: str = "structure",
        target_rows: list[int] | tuple[int, ...] | None = None,
        train_rows: list[int] | tuple[int, ...] | None = None,
    ) -> "PrepropBlockSpec":
        return cls(name=name, kind="structure", target_rows=_rows(target_rows), train_rows=_rows(train_rows))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "relation": _relation_payload(self.relation),
            "path_schema": [_relation_payload(relation) for relation in self.path_schema],
            "target_rows": self.target_rows,
            "train_rows": self.train_rows,
            "top_k": int(self.top_k),
            "metadata": self.metadata,
        }

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
