from __future__ import annotations

from pathlib import Path
from typing import Any

from shadow_hgc.ultra.papers100m_memmap import read_json


class Papers100MCacheContext:
    def __init__(self, cache_root: str | Path, mode: str = "r", *, selection_policy: str = "stt_ratio_v2", seed: int = 42):
        self.cache_root = Path(cache_root)
        self.mode = str(mode)
        self.selection_policy = str(selection_policy)
        self.seed = int(seed)
        self.manifest = self._read("manifest.json")
        self.raw = self._read("raw/raw_manifest.json")
        self.graph = self._read_optional("graph/edge_slice_manifest.json")
        self.sft = self._read_optional("sft/sft_manifest.json")
        self.teacher = self._read_optional("teacher/teacher_cache_manifest.json")
        self.bank = self._read_bank_manifest()

    def _read(self, rel_path: str) -> dict[str, Any]:
        return read_json(self.cache_root / rel_path)

    def _read_optional(self, rel_path: str) -> dict[str, Any]:
        path = self.cache_root / rel_path
        return read_json(path) if path.exists() else {}

    def _read_bank_manifest(self) -> dict[str, Any]:
        rel = f"selection_bank/policy={self.selection_policy}_seed{self.seed}/bank_manifest.json"
        path = self.cache_root / rel
        if path.exists():
            return read_json(path)
        candidates = sorted((self.cache_root / "selection_bank").glob(f"policy={self.selection_policy}_seed*/bank_manifest.json"))
        if len(candidates) == 1:
            return read_json(candidates[0])
        return {}

    def cache_ids(self) -> dict[str, str]:
        return {
            "cache_build_id": str(self.manifest.get("cache_build_id", "")),
            "edge_slice_cache_id": str(self.graph.get("edge_slice_cache_id", self.graph.get("edge_cache_id", ""))),
            "sft_cache_id": str(self.sft.get("sft_cache_id", "")),
            "teacher_cache_id": str(self.teacher.get("teacher_cache_id", "")),
            "selection_bank_id": str(self.bank.get("selection_bank_id", "")),
        }

    def assert_ready(self, required: list[str]) -> None:
        missing = []
        for name in required:
            if name == "manifest" and not self.manifest:
                missing.append(name)
            elif name == "edge_cache" and not self.graph:
                missing.append(name)
            elif name == "sft_cache" and not self.sft:
                missing.append(name)
            elif name == "teacher_cache" and not self.teacher:
                missing.append(name)
            elif name == "selection_bank" and not self.bank:
                missing.append(name)
        if missing:
            raise FileNotFoundError(f"T35 cache context missing required stages: {missing}")
