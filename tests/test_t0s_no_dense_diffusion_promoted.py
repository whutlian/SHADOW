from __future__ import annotations

from shadow_hgc.fullgraph.t0s_gates import evaluate_t0s_row


def test_t0s_promoted_row_rejects_diffusion_dense_p2_and_graph_backprop():
    row = {
        "dataset": "ogbn-products",
        "accuracy": 0.8,
        "uses_diffusion": True,
        "uses_dense_p2": True,
        "uses_full_graph_backprop": True,
        "train_label_only": True,
        "full_edge_scans": 2,
        "cache_bytes": 128,
        "peak_cpu_ram_gb": 1.0,
        "peak_gpu_ram_gb": 0.0,
    }

    result = evaluate_t0s_row(row)

    assert result["gate_scalability_passed"] is False
    assert "uses_diffusion" in result["blocked_reason"]
    assert "uses_dense_p2" in result["blocked_reason"]
    assert "uses_full_graph_backprop" in result["blocked_reason"]


def test_t0s_accuracy_gate_is_dataset_specific():
    row = {
        "dataset": "acm",
        "accuracy": 0.931,
        "uses_diffusion": False,
        "uses_dense_p2": False,
        "uses_full_graph_backprop": False,
        "train_label_only": True,
        "full_edge_scans": 2,
        "cache_bytes": 128,
        "peak_cpu_ram_gb": 1.0,
        "peak_gpu_ram_gb": 0.0,
    }

    result = evaluate_t0s_row(row)

    assert result["gate_acc"] == 0.93
    assert result["gate_acc_passed"] is True
    assert result["gate_scalability_passed"] is True
