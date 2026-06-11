from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RawTextLoadResult:
    available: bool
    text_map: dict[int, str]
    source_path: str
    failure_reason: str
    actionable_message: str
    precomputed_embedding_path: str = ""


def _read_jsonl(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            node_id = int(item.get("node_id", item.get("id", item.get("paper_id", len(out)))))
            title = str(item.get("title", ""))
            abstract = str(item.get("abstract", ""))
            out[node_id] = (title + "\n" + abstract).strip()
    return out


def _read_csv(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            node_id = int(row.get("node_id", row.get("id", idx)))
            title = str(row.get("title", ""))
            abstract = str(row.get("abstract", ""))
            text = str(row.get("text", "")).strip() or (title + "\n" + abstract).strip()
            out[node_id] = text
    return out


def load_arxiv_raw_text_map(
    *,
    search_paths: Iterable[str | Path] | None = None,
    precomputed_embedding_path: str | Path | None = None,
) -> RawTextLoadResult:
    if precomputed_embedding_path:
        path = Path(precomputed_embedding_path)
        if path.exists():
            return RawTextLoadResult(
                available=True,
                text_map={},
                source_path=str(path),
                failure_reason="",
                actionable_message="Using precomputed semantic embedding cache.",
                precomputed_embedding_path=str(path),
            )
    for raw in search_paths or []:
        path = Path(raw)
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            text_map = _read_jsonl(path)
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            text_map = {int(k): str(v) for k, v in payload.items()}
        elif path.suffix == ".csv":
            text_map = _read_csv(path)
        else:
            continue
        if text_map:
            return RawTextLoadResult(True, text_map, str(path), "", "Loaded local arxiv raw text map.")
    return RawTextLoadResult(
        available=False,
        text_map={},
        source_path="",
        failure_reason="raw_text_missing",
        actionable_message=(
            "Provide a local title/abstract JSONL/CSV via --raw-text-path or a precomputed memmap via "
            "--use-precomputed-semantic-features."
        ),
    )


def write_semantic_cache_manifest(
    cache_dir: str | Path,
    *,
    model_name: str,
    embedding_path: str,
    num_nodes: int,
    feature_dim: int,
    dtype: str,
    cache_bytes: int,
    node_ordering: str = "ogb_node_id",
    text_fields: list[str] | tuple[str, ...] = ("title", "abstract"),
) -> Path:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_name": model_name,
        "embedding_path": embedding_path,
        "shape": [int(num_nodes), int(feature_dim)],
        "dtype": dtype,
        "cache_bytes": int(cache_bytes),
        "node_ordering": node_ordering,
        "text_fields": list(text_fields),
    }
    path = root / "semantic_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_semantic_cache_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def semantic_flags(
    *,
    model_name: str,
    feature_dim: int,
    cache_bytes: int,
    raw_text_encoded: bool,
    encode_time: float,
) -> dict[str, Any]:
    return {
        "uses_external_text_features": True,
        "uses_raw_text": bool(raw_text_encoded),
        "uses_lm_encoder": True,
        "semantic_lm_model": model_name,
        "semantic_feature_dim": int(feature_dim),
        "semantic_cache_bytes": int(cache_bytes),
        "semantic_encode_time": float(encode_time),
        "promotion_track": "sota_chase",
    }
