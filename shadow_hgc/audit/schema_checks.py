from __future__ import annotations

import json
from typing import Any


def coerce_list(value: Any) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return list(parsed)
        return [parsed] if parsed not in (None, "") else []
    return [value]


def coerce_dict(value: Any) -> dict:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def require_nonempty_feature_blocks(row: dict, *, required_prefix: str) -> dict:
    """Check that a model row logs non-empty feature blocks and positive dims."""

    block_key = f"{required_prefix}_blocks"
    blocks = coerce_list(row.get(block_key))
    block_dims = coerce_dict(row.get("block_dims"))
    if not block_dims:
        block_dims = coerce_dict(row.get(f"{required_prefix}_block_dims"))
    feature_blocks = coerce_list(row.get("feature_blocks"))
    reasons: list[str] = []
    if not blocks:
        reasons.append(f"{block_key}_empty")
    for block in blocks:
        dim = block_dims.get(str(block), block_dims.get(block))
        if dim is None and feature_blocks:
            continue
        try:
            if dim is not None and int(dim) <= 0:
                reasons.append(f"{block_key}_dim_nonpositive:{block}")
        except (TypeError, ValueError):
            reasons.append(f"{block_key}_dim_invalid:{block}")
    if blocks and block_dims:
        missing = [str(block) for block in blocks if str(block) not in {str(key) for key in block_dims}]
        if missing:
            reasons.append(f"{block_key}_dims_missing:{','.join(missing)}")
    return {"valid": not reasons, "reasons": reasons, "warnings": []}

