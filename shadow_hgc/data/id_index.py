from __future__ import annotations

from dataclasses import dataclass

import torch


def select_id_index_mode(*, num_nodes: int, dense_map_budget_bytes: int) -> str:
    """Choose the id index representation without allocating it."""

    return "dense_int32" if int(num_nodes) * 4 <= int(dense_map_budget_bytes) else "sorted_search"


@dataclass(frozen=True)
class IdIndex:
    """Map global node ids to compact positions, returning -1 for misses."""

    mode: str
    num_nodes: int
    _dense_map: torch.Tensor | None = None
    _sorted_ids: torch.Tensor | None = None
    _sorted_positions: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        ids: torch.Tensor,
        *,
        num_nodes: int,
        dense_map_budget_bytes: int = 64 * 1024 * 1024,
    ) -> "IdIndex":
        ids = ids.detach().to(torch.long).cpu().flatten()
        num_nodes = int(num_nodes)
        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= num_nodes):
            raise ValueError("index ids must be within [0, num_nodes)")

        sorted_ids, order = torch.sort(ids)
        if sorted_ids.numel() > 1 and bool((sorted_ids[1:] == sorted_ids[:-1]).any()):
            raise ValueError("IdIndex ids must not contain duplicate entries")

        mode = select_id_index_mode(
            num_nodes=num_nodes,
            dense_map_budget_bytes=dense_map_budget_bytes,
        )
        if mode == "dense_int32":
            dense_map = torch.full((num_nodes,), -1, dtype=torch.int32)
            if ids.numel():
                dense_map[ids] = torch.arange(ids.numel(), dtype=torch.int32)
            return cls(mode=mode, num_nodes=num_nodes, _dense_map=dense_map)

        return cls(
            mode=mode,
            num_nodes=num_nodes,
            _sorted_ids=sorted_ids,
            _sorted_positions=order.to(torch.long),
        )

    @property
    def storage_nbytes(self) -> int:
        if self._dense_map is not None:
            return int(self._dense_map.numel() * self._dense_map.element_size())
        total = 0
        if self._sorted_ids is not None:
            total += int(self._sorted_ids.numel() * self._sorted_ids.element_size())
        if self._sorted_positions is not None:
            total += int(self._sorted_positions.numel() * self._sorted_positions.element_size())
        return total

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """Return compact positions for ids, with -1 for missing or out-of-range ids."""

        original_shape = ids.shape
        query = ids.detach().to(torch.long).cpu().flatten()
        result = torch.full((query.numel(),), -1, dtype=torch.long)
        if query.numel() == 0:
            return result.reshape(original_shape)

        valid = (query >= 0) & (query < self.num_nodes)
        if not bool(valid.any()):
            return result.reshape(original_shape)

        if self._dense_map is not None:
            result[valid] = self._dense_map[query[valid]].to(torch.long)
            return result.reshape(original_shape)

        if self._sorted_ids is None or self._sorted_positions is None or self._sorted_ids.numel() == 0:
            return result.reshape(original_shape)

        valid_query = query[valid]
        insertion = torch.searchsorted(self._sorted_ids, valid_query)
        in_bounds = insertion < self._sorted_ids.numel()
        matched = torch.zeros_like(in_bounds, dtype=torch.bool)
        if bool(in_bounds.any()):
            matched[in_bounds] = self._sorted_ids[insertion[in_bounds]] == valid_query[in_bounds]

        valid_positions = torch.full((valid_query.numel(),), -1, dtype=torch.long)
        if bool(matched.any()):
            valid_positions[matched] = self._sorted_positions[insertion[matched]]
        result[valid] = valid_positions
        return result.reshape(original_shape)
