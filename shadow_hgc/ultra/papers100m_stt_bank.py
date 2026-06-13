from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shadow_hgc.ultra.papers100m_memmap import directory_bytes, read_json, stable_hash, utc_now, write_json
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_teacher import load_teacher_topk_cache


def _budget(total: int) -> dict[str, int]:
    weights = {"core": 0.45, "boundary": 0.20, "rare": 0.15, "prior_repair": 0.10, "hard_anchor": 0.10}
    alloc = {key: int(math.floor(total * value)) for key, value in weights.items()}
    for key in sorted(weights, key=weights.get, reverse=True):
        if sum(alloc.values()) >= total:
            break
        alloc[key] += 1
    return alloc


def _push(heap: list[tuple[float, int]], item: tuple[float, int], limit: int) -> None:
    if limit <= 0:
        return
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


@dataclass
class SelectionBank:
    root: Path
    manifest: dict[str, Any]
    global_queue: np.memmap

    def select_prefix(self, ratio: float, *, full_node_denominator: int | None = None) -> np.ndarray:
        denom = int(full_node_denominator or self.manifest.get("full_node_ratio_denominator", self.manifest["target_universe_size"]))
        count = max(1, int(round(float(ratio) * denom)))
        count = min(count, int(self.global_queue.shape[0]))
        return np.asarray(self.global_queue[:count], dtype=np.uint32)


class StreamingSTTBankBuilder:
    def __init__(
        self,
        cache_context: Papers100MCacheContext,
        policy: str,
        seed: int,
        max_ratio: float,
        *,
        chunk_size: int = 262_144,
        nested_selection: bool = True,
    ) -> None:
        self.ctx = cache_context
        self.policy = str(policy)
        self.seed = int(seed)
        self.max_ratio = float(max_ratio)
        self.chunk_size = int(chunk_size)
        self.nested_selection = bool(nested_selection)

    def build_bank(self, *, force: bool = False) -> dict[str, Any]:
        bank_dir = self.ctx.cache_root / "selection_bank" / f"policy={self.policy}_seed{self.seed}"
        manifest_path = bank_dir / "bank_manifest.json"
        if manifest_path.exists() and not force:
            return read_json(manifest_path)
        self.ctx.assert_ready(["manifest", "sft_cache", "teacher_cache"])
        started = time.perf_counter()
        bank_dir.mkdir(parents=True, exist_ok=True)
        target_size = int(self.ctx.manifest["target_universe_size"])
        denom = int(self.ctx.manifest["num_nodes"])
        max_rows = min(target_size, max(1, int(round(self.max_ratio * denom))))
        quotas = _budget(max_rows)
        heaps: dict[str, list[tuple[float, int]]] = {key: [] for key in quotas}
        scores = np.memmap(bank_dir / "scores.fp32.memmap", mode="w+", dtype=np.float32, shape=(target_size,))
        teacher = load_teacher_topk_cache(self.ctx.cache_root)
        degree = np.memmap(self.ctx.cache_root / "sft" / "degree_target.fp16.memmap", mode="r", dtype=np.float16, shape=(target_size, 2))
        train_local = set(np.asarray(np.memmap(self.ctx.cache_root / "raw" / "train_local_idx.u32.memmap", mode="r", dtype=np.uint32, shape=(int(self.ctx.manifest["train_size"]),)), dtype=np.int64).tolist())
        rng = np.random.default_rng(self.seed)
        for start in range(0, target_size, self.chunk_size):
            stop = min(start + self.chunk_size, target_size)
            ids = np.asarray(teacher.topk_class_ids[start:stop], dtype=np.int64)
            probs = np.asarray(teacher.topk_probs[start:stop], dtype=np.float32)
            entropy = np.asarray(teacher.entropy[start:stop], dtype=np.float32)
            margin = np.asarray(teacher.margin[start:stop], dtype=np.float32)
            deg = np.asarray(degree[start:stop], dtype=np.float32).sum(axis=1)
            confidence = probs[:, 0]
            noise = rng.random(stop - start).astype(np.float32) * 1e-6
            global_score = confidence + 0.05 * deg - 0.01 * entropy + noise
            scores[start:stop] = global_score
            for offset, local_id in enumerate(range(start, stop)):
                _push(heaps["core"], (float(confidence[offset] + noise[offset]), local_id), quotas["core"])
                _push(heaps["boundary"], (float(entropy[offset] - margin[offset] + noise[offset]), local_id), quotas["boundary"])
                _push(heaps["rare"], (float(-deg[offset] + 0.01 * ids[offset, 0] + noise[offset]), local_id), quotas["rare"])
                _push(heaps["prior_repair"], (float((1.0 - confidence[offset]) + noise[offset]), local_id), quotas["prior_repair"])
                if local_id in train_local:
                    _push(heaps["hard_anchor"], (float(confidence[offset] + noise[offset]), local_id), quotas["hard_anchor"])
        scores.flush()

        selected: list[int] = []
        seen: set[int] = set()
        bucket_arrays: dict[str, np.ndarray] = {}
        for bucket in ("core", "boundary", "rare", "prior_repair", "hard_anchor"):
            ordered = [local_id for _score, local_id in sorted(heaps[bucket], reverse=True)]
            bucket_arrays[bucket] = np.asarray(ordered, dtype=np.uint32)
            for local_id in ordered:
                if local_id not in seen and len(selected) < max_rows:
                    selected.append(local_id)
                    seen.add(local_id)
        if len(selected) < max_rows:
            order = np.argsort(-np.asarray(scores, dtype=np.float32))
            for local_id in order:
                value = int(local_id)
                if value not in seen:
                    selected.append(value)
                    seen.add(value)
                    if len(selected) >= max_rows:
                        break
        for bucket, values in bucket_arrays.items():
            queue_path = bank_dir / f"{bucket}.queue.u32.memmap"
            if values.size == 0:
                queue_path.write_bytes(b"")
            else:
                mm = np.memmap(queue_path, mode="w+", dtype=np.uint32, shape=(values.size,))
                mm[:] = values
                mm.flush()
                del mm
        global_queue = np.memmap(bank_dir / "global.queue.u32.memmap", mode="w+", dtype=np.uint32, shape=(len(selected),))
        global_queue[:] = np.asarray(selected, dtype=np.uint32)
        global_queue.flush()
        del global_queue
        manifest = {
            "dataset_name": self.ctx.manifest["dataset_name"],
            "selection_policy": self.policy,
            "seed": self.seed,
            "max_ratio": self.max_ratio,
            "full_node_ratio_denominator": denom,
            "target_universe_size": target_size,
            "selected_max_rows": int(len(selected)),
            "nested_selection": self.nested_selection,
            "bank_build_count": 1,
            "bucket_core_count": int(bucket_arrays["core"].size),
            "bucket_boundary_count": int(bucket_arrays["boundary"].size),
            "bucket_rare_count": int(bucket_arrays["rare"].size),
            "bucket_prior_repair_count": int(bucket_arrays["prior_repair"].size),
            "bucket_hard_anchor_count": int(bucket_arrays["hard_anchor"].size),
            "selection_bank_time": float(time.perf_counter() - started),
            "selection_bank_bytes": directory_bytes(bank_dir),
            "parent_cache_ids": self.ctx.cache_ids(),
            "created_at": utc_now(),
        }
        manifest["selection_bank_id"] = stable_hash(
            {"policy": self.policy, "seed": self.seed, "max_ratio": self.max_ratio, "parents": manifest["parent_cache_ids"], "rows": len(selected)}
        )
        write_json(manifest_path, manifest)
        return manifest


def load_selection_bank(cache_root: str | Path, *, policy: str = "stt_ratio_v2", seed: int = 42) -> SelectionBank:
    root = Path(cache_root) / "selection_bank" / f"policy={policy}_seed{seed}"
    if not (root / "bank_manifest.json").exists():
        candidates = sorted((Path(cache_root) / "selection_bank").glob(f"policy={policy}_seed*/bank_manifest.json"))
        if len(candidates) == 1:
            root = candidates[0].parent
    manifest = read_json(root / "bank_manifest.json")
    queue = np.memmap(root / "global.queue.u32.memmap", mode="r", dtype=np.uint32, shape=(int(manifest["selected_max_rows"]),))
    return SelectionBank(root=root, manifest=manifest, global_queue=queue)
