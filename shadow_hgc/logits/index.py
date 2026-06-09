from __future__ import annotations

from pathlib import Path
from typing import Any

from shadow_hgc.logits.replay import metadata_for_cache


def discover_logit_caches(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root)
    if not base.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(base.glob("**/meta.json")):
        meta = metadata_for_cache(path.parent)
        rows.append({**meta, "cache_path": str(path.parent)})
    return rows
