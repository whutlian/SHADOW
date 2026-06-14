from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import json
from typing import Any, Iterable


@dataclass(frozen=True)
class CandidateStats:
    node_id: int
    coverage_priority: float = 0.0
    hard_anchor_quality: float = 0.0
    teacher_soft_value: float = 0.0
    boundary_value: float = 0.0
    rare_value: float = 0.0
    redundancy_penalty: float = 0.0


def score_candidate(candidate: CandidateStats, weights: dict[str, float]) -> float:
    return (
        float(weights.get("coverage", 0.0)) * candidate.coverage_priority
        + float(weights.get("hard", 0.0)) * candidate.hard_anchor_quality
        + float(weights.get("soft", 0.0)) * candidate.teacher_soft_value
        + float(weights.get("boundary", 0.0)) * candidate.boundary_value
        + float(weights.get("rare", 0.0)) * candidate.rare_value
        - float(weights.get("diversity", 0.0)) * candidate.redundancy_penalty
    )


@dataclass
class StreamingUnifiedReservoir:
    capacity: int
    weights: dict[str, float]
    _heap: list[tuple[float, int, CandidateStats]] = field(default_factory=list)

    def add(self, candidate: CandidateStats) -> None:
        item = (score_candidate(candidate, self.weights), int(candidate.node_id), candidate)
        if len(self._heap) < int(self.capacity):
            heapq.heappush(self._heap, item)
            return
        if item > self._heap[0]:
            heapq.heapreplace(self._heap, item)

    def add_many(self, candidates: Iterable[CandidateStats]) -> None:
        for candidate in candidates:
            self.add(candidate)

    def selected_ids(self) -> list[int]:
        return [item[2].node_id for item in sorted(self._heap, reverse=True)]


def teacher_free_selection_signature(
    *,
    candidate_ids: Iterable[int],
    train_labels: dict[int, int] | None = None,
    valid_labels: dict[int, int] | None = None,
    test_labels: dict[int, int] | None = None,
    extra_teacher_free_stats: dict[str, Any] | None = None,
) -> str:
    del valid_labels, test_labels
    payload = {
        "candidate_ids": sorted(int(value) for value in candidate_ids),
        "train_labels": sorted((int(key), int(value)) for key, value in (train_labels or {}).items()),
        "extra_teacher_free_stats": extra_teacher_free_stats or {},
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
