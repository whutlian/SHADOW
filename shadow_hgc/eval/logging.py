from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def attach_run_metadata(payload: dict[str, Any], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    enriched = dict(payload)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    enriched.setdefault("generated_at", generated_at)
    enriched.setdefault("git_commit", _git_commit())
    enriched.setdefault("config_hash", config_hash(config if config is not None else enriched))
    base = enriched.get("method") or enriched.get("baseline") or "run"
    dataset = enriched.get("dataset", "dataset")
    seed = enriched.get("seed", "na")
    enriched.setdefault("run_id", f"{dataset}-{base}-seed{seed}-{enriched['config_hash'][:8]}")
    return enriched


def write_json_summary(path: str | Path, payload: dict[str, Any], *, config: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(attach_run_metadata(payload, config=config)), indent=2), encoding="utf-8")
