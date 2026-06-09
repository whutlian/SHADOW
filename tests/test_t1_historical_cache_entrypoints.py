from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from scripts.run_t1_generate_safe_logit_caches import _existing_cache_row, _model_config_payload
from scripts.t1_safe_common import save_cache_for_graph
from shadow_hgc.data.loaders import build_toy_graph
from shadow_hgc.logits import load_logits_cache
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment
from shadow_hgc.train.sehgnn_lite_target import SeHGNNTargetRun


def test_sehgnn_target_run_exposes_logits_field():
    logits = torch.randn(4, 3)
    run = SeHGNNTargetRun(summary={"accuracy": 0.5}, blocks={"self": torch.randn(4, 2)}, logits=logits)

    assert run.logits is logits


def test_pipeline_experiment_accepts_return_logits_flag():
    signature = inspect.signature(run_shadow_hgc_experiment)

    assert "return_logits" in signature.parameters


def test_pipeline_return_logits_emits_all_target_logits(tmp_path):
    graph = build_toy_graph(seed=42)

    summary = run_shadow_hgc_experiment(
        graph,
        output_path=tmp_path / "toy.json",
        seed=42,
        epochs=1,
        budget_mode="count",
        target_budget=2,
        feature_dim=4,
        projection_type="raw",
        hidden_dim=8,
        demand_edge_chunk_size=32,
        inference_edge_chunk_size=32,
        return_logits=True,
    )

    logits = summary["_target_logits"]
    assert logits.shape == (graph.num_nodes[graph.target_type], summary["num_classes_global"])
    assert "accuracy" in summary


def test_compiled_pipeline_return_logits_emits_split_assembled_target_logits(tmp_path):
    graph = build_toy_graph(seed=7)

    summary = run_shadow_hgc_experiment(
        graph,
        output_path=tmp_path / "toy_compiled.json",
        seed=7,
        epochs=1,
        budget_mode="count",
        target_budget=2,
        feature_dim=4,
        projection_type="raw",
        model_type="shadow_fusion",
        feature_mode="label_affinity",
        label_affinity=True,
        label_affinity_mode="target_target",
        label_affinity_self_exclude=True,
        compiled_head=True,
        compiled_demand_source="shadow_reconstructed",
        compiled_block_stats_source="train_full_demand_table",
        hidden_dim=8,
        compiled_hidden_dim=8,
        demand_edge_chunk_size=32,
        inference_edge_chunk_size=32,
        inference_dst_chunk_size=4,
        return_logits=True,
    )

    logits = summary["_target_logits"]
    assert logits.shape == (graph.num_nodes[graph.target_type], summary["num_classes_global"])
    covered = torch.cat([graph.train_idx, graph.val_idx, graph.test_idx])
    assert torch.any(logits[covered].abs() > 0)


def test_compiled_pipeline_return_logits_is_disabled_for_ultra_scale(tmp_path):
    graph = replace(build_toy_graph(seed=7), dataset_name="mag240m")

    with pytest.raises(ValueError, match="disabled for ultra-scale"):
        run_shadow_hgc_experiment(
            graph,
            output_path=tmp_path / "toy_compiled.json",
            seed=7,
            epochs=1,
            budget_mode="count",
            target_budget=2,
            feature_dim=4,
            projection_type="raw",
            compiled_head=True,
            feature_mode="label_affinity",
            label_affinity=True,
            return_logits=True,
        )


def test_save_cache_for_graph_writes_replay_metadata(tmp_path):
    graph = SimpleNamespace(
        dataset_name="toy",
        target_type="paper",
        labels=torch.tensor([0, 1, 2, 1]),
        train_idx=torch.tensor([0, 1]),
        test_idx=torch.tensor([3]),
        num_nodes={"paper": 4},
    )
    logits = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 5.0, 0.0],
        ]
    )
    base = {
        "dataset": "toy",
        "base_variant": "historical row",
        "cache_variant": "historical_cache",
        "expected_acc": 1.0,
    }
    path = save_cache_for_graph(
        root=tmp_path,
        graph=graph,
        logits=logits,
        train_rows=graph.train_idx,
        valid_rows=torch.tensor([2]),
        base=base,
        seed=42,
        role="historical_replay",
        metrics={"test": {"accuracy": 1.0, "macro_f1": 1.0, "predicted_class_count": 1}},
        dtype="float32",
    )

    loaded = load_logits_cache(path)
    assert loaded.all_target_logits is not None
    assert loaded.all_target_logits.shape == (4, 3)
    assert loaded.meta.variant == "historical row"
    metadata = (path / "metadata.json").read_text(encoding="utf-8")
    assert '"cache_role": "historical_replay"' in metadata
    assert '"historical_expected_acc": 1.0' in metadata


def test_existing_cache_row_rejects_wrong_model_config_hash(tmp_path):
    graph = SimpleNamespace(
        dataset_name="toy",
        target_type="paper",
        labels=torch.tensor([0, 1, 2, 1]),
        train_idx=torch.tensor([0, 1]),
        test_idx=torch.tensor([3]),
        num_nodes={"paper": 4},
    )
    logits = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 5.0, 0.0],
        ]
    )
    base = {
        "dataset": "toy",
        "base_variant": "historical row",
        "cache_variant": "historical_cache",
        "expected_acc": 1.0,
        "macro_f1": 1.0,
    }
    args = SimpleNamespace(cache_root=str(tmp_path), seed=42)
    metrics = {"test": {"accuracy": 1.0, "macro_f1": 1.0, "predicted_class_count": 1}}

    save_cache_for_graph(
        root=tmp_path,
        graph=graph,
        logits=logits,
        train_rows=graph.train_idx,
        valid_rows=torch.tensor([2]),
        base=base,
        seed=42,
        role="historical_replay",
        metrics=metrics,
        dtype="float32",
        model_config_payload={"wrong": True},
    )
    assert _existing_cache_row(base, args, gate_same_as_replay=True) is None

    save_cache_for_graph(
        root=tmp_path,
        graph=graph,
        logits=logits,
        train_rows=graph.train_idx,
        valid_rows=torch.tensor([2]),
        base=base,
        seed=42,
        role="historical_replay",
        metrics=metrics,
        dtype="float32",
        model_config_payload=_model_config_payload(base, args, "historical_replay"),
    )
    assert _existing_cache_row(base, args, gate_same_as_replay=True)["dataset"] == "toy"
