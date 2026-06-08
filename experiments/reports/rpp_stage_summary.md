# Shadow-HGC R++ Stage Summary

Date: 2026-06-08

Scope: this report covers only the R++ performance and scalability sprint. The frozen Shadow-HGC-R-1 default path remains unchanged unless R++ flags or scripts are used.

## What Changed In R++

Code modules added:

| Area | Files |
|---|---|
| Safe model construction | `shadow_hgc/models/factory.py` |
| Late fusion classifier | `shadow_hgc/models/shadow_fusion.py` |
| Streaming diffusion | `shadow_hgc/features/streaming_diffusion.py` |
| Feature block normalization | `shadow_hgc/features/block_norm.py` |
| Chunked shadow assignment | `shadow_hgc/shadows/assign.py` |
| Global shadow cap | `shadow_hgc/shadows/adaptive.py` |
| Pipeline logging/integration | `shadow_hgc/pipeline/core.py` |

Experiment scripts added:

| Script | Output table | Output report |
|---|---|---|
| `scripts/run_rpp_imdb_rescue.py` | `experiments/tables/imdb_rpp_rescue_seed42.csv` | `experiments/reports/imdb_rpp_rescue_summary.md` |
| `scripts/run_rpp_arxiv_refine.py` | `experiments/tables/arxiv_rpp_refine_seed42.csv` | `experiments/reports/arxiv_rpp_refine_summary.md` |
| `scripts/run_rpp_products_streaming_diffusion.py` | `experiments/tables/products_streaming_diffusion_seed42.csv` | `experiments/reports/products_streaming_diffusion_summary.md` |
| `scripts/run_rpp_small_nonregression.py` | `experiments/tables/small_rpp_nonregression_seed42.csv` | `experiments/reports/small_rpp_nonregression_summary.md` |
| `scripts/run_rpp_stage.py` | orchestrates the four scripts | N/A |

Tests added:

| Requirement | Test files |
|---|---|
| No final ReLU logits | `tests/test_no_final_relu_logits.py` |
| Streaming diffusion | `tests/test_streaming_diffusion_matches_dense.py`, `tests/test_streaming_diffusion_memory_guard.py` |
| Block normalization | `tests/test_block_norm_no_nan.py`, `tests/test_block_norm_preserves_shapes.py` |
| Shadow fusion | `tests/test_shadow_fusion_schema_preservation.py`, `tests/test_shadow_fusion_gate_logging.py`, `tests/test_shadow_fusion_chunked_inference.py` |
| Ratio cap and chunked assignment | `tests/test_total_ratio_global_cap.py`, `tests/test_adaptive_shadow_budget_sum.py`, `tests/test_chunked_shadow_assignment.py` |

## Code Correctness Gate

Latest full test command:

```powershell
C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests -q
```

Result:

```text
100 passed in 73.86s
```

The R++ model factory rejects unsafe final ReLU logits. `relation_linear` final logits use `activation=None`; `relation_mlp` uses ReLU only in the hidden layer; `shadow_fusion` returns raw logits.

## Experiment Matrix

All experiments used seed 42 only.

| Group | Completed rows | Failed rows | Status |
|---|---:|---:|---|
| IMDB rescue v2 | 24 | 0 | completed |
| ogbn-arxiv refine | 12 | 0 | completed |
| ogbn-products streaming diffusion | 12 | 12 | diagnostic only, products chunk path dropped |
| ACM/DBLP non-regression | 6 | 0 | completed |

## Best Rows

| Dataset | R++ best variant | Ratio | Accuracy | Macro-F1 | Predicted classes | Previous R+ best |
|---|---|---:|---:|---:|---:|---:|
| IMDB | `full_rplus_shadow_fusion_adaptive_b`, class-balanced | 2.5% | 0.4076 | 0.3842 | 5 | 0.3810 |
| ogbn-arxiv | `diffusion_X0X1X2_highpass_blocknorm_shadow_fusion` | 12.0% | 0.6172 | 0.4147 | 40 | 0.5369 |
| ogbn-products | `base`, shadow-fusion, sqrt-weighted | 12.0% | 0.6689 | 0.3080 | 41 | 0.5891 |
| ACM | `current_best` | 9.6% | 0.8432 | 0.8462 | 3 | 0.8432 |
| DBLP | `current_best` | 6.5% | 0.8370 | 0.8299 | 4 | 0.8370 |

## Detailed Results By Group

### IMDB Rescue V2

CSV: `experiments/tables/imdb_rpp_rescue_seed42.csv`

Best row:

| Variant | Loss | Ratio | Accuracy | Macro-F1 | Predicted classes |
|---|---|---:|---:|---:|---:|
| `full_rplus_shadow_fusion_adaptive_b` | class-balanced | 2.5% | 0.4076 | 0.3842 | 5 |

Interpretation:

- Gate passed: accuracy is above 0.405, macro-F1 is above 0.36, and all five classes are predicted.
- Shadow fusion and class-balanced loss are useful on IMDB.
- The adaptive-b and non-adaptive shadow-fusion rows tied at the best point, so adaptive-b is not the source of the improvement by itself.

### ogbn-arxiv R++ Refinement

CSV: `experiments/tables/arxiv_rpp_refine_seed42.csv`

Best rows:

| Variant | Ratio | Accuracy | Macro-F1 | Predicted classes |
|---|---:|---:|---:|---:|
| `diffusion_X0X1X2_highpass_blocknorm_shadow_fusion` | 12.0% | 0.6172 | 0.4147 | 40 |
| `diffusion_X0X1X2_highpass_blocknorm_shadow_fusion` | 6.0% | 0.6029 | 0.3778 | 40 |
| `diffusion_X0X1X2_highpass_shadow_fusion` | 12.0% | 0.5989 | 0.3998 | 40 |

Interpretation:

- Gate passed: best accuracy is above 0.56 and macro-F1 is above 0.35.
- Block normalization plus shadow fusion is the strongest R++ combination on arxiv.
- Relation-linear no-final-ReLU is safe but weaker than shadow fusion for this stage.

### ogbn-products Streaming Diffusion

CSV: `experiments/tables/products_streaming_diffusion_seed42.csv`

Best completed row:

| Variant | Loss | Ratio | Accuracy | Macro-F1 | Predicted classes |
|---|---|---:|---:|---:|---:|
| `base` | sqrt-weighted | 12.0% | 0.6689 | 0.3080 | 41 |

Drop Decision:

- The second-stage destination/edge chunking fixed the immediate inference OOM, but the full products run took about 102 minutes.
- Completed `streaming_diffusion_X0X1X2` rows match the `base` rows exactly because the memmap diffusion blocks were precomputed but not yet wired into the products lazy `phi` feature provider.
- `streaming_diffusion_X0X1` and `streaming_diffusion_X0X1X2_highpass` rows failed with Windows memmap path handling (`Errno 22`).
- This path is dropped from the recommended R++ configuration. Keep it only as a scalability diagnostic.

Interpretation:

- Products accuracy target is met by `base + shadow_fusion` at 12.0%, but this is not evidence of streaming diffusion improving products.
- Next products work should focus on a true lazy block provider for `phi` before any additional products grid.

### ACM / DBLP Non-Regression

CSV: `experiments/tables/small_rpp_nonregression_seed42.csv`

Rows:

| Dataset | Variant | Ratio | Accuracy | Macro-F1 | Predicted classes |
|---|---|---:|---:|---:|---:|
| ACM | `current_best` | 9.6% | 0.8432 | 0.8462 | 3 |
| ACM | `shadow_fusion_blocknorm` | 9.6% | 0.8187 | 0.8192 | 3 |
| DBLP | `current_best` | 0.5% | 0.8282 | 0.8214 | 4 |
| DBLP | `current_best` | 6.5% | 0.8370 | 0.8299 | 4 |
| DBLP | `shadow_fusion_blocknorm` | 0.5% | 0.7158 | 0.7127 | 4 |
| DBLP | `shadow_fusion_blocknorm` | 6.5% | 0.7092 | 0.7033 | 4 |

Interpretation:

- Current-best R+ settings preserve the expected ACM and DBLP results.
- Shadow-fusion blocknorm is not suitable for ACM/DBLP in its current form and should not be promoted as the small-dataset default.

## Compression And Resource Accounting

Every completed JSON row now logs:

```json
{
  "requested_target_ratio": "...",
  "effective_target_ratio": "...",
  "shadow_node_ratio": "...",
  "total_condensed_node_ratio": "...",
  "total_condensed_edge_ratio": "...",
  "byte_size_compression": "...",
  "effective_M_tau": "...",
  "shadow_nodes_total": "...",
  "condensed_nodes_total": "...",
  "condensed_edges_total": "..."
}
```

The products best completed row at 12.0% has total condensed node ratio about 0.0140 and byte-size compression about 0.0036. The arxiv and IMDB detailed ratio/accounting fields are in their CSV and JSON rows.

## Acceptance Gates

| Gate | Result | Evidence |
|---|---|---|
| Code correctness | PASS | `100 passed in 73.86s`; no-final-ReLU tests pass |
| Products scalability | PARTIAL / DROPPED | Chunked inference avoids the earlier large allocation, but products streaming feature path is not promoted |
| Medium accuracy improvement | PASS | arxiv best 0.6172 > 0.56; products base shadow-fusion 0.6689 > 0.5891 |
| IMDB rescue | PASS | 0.4076 accuracy, 0.3842 macro-F1, 5 predicted classes |
| Small non-regression | PARTIAL | ACM/DBLP current-best hold; shadow_fusion_blocknorm regresses and is dropped |
| Fair compression accounting | PASS | ratio/edge/byte fields are logged in completed rows |

## Recommended R++ Settings From This Stage

Promote for follow-up experiments:

- IMDB: `shadow_fusion`, class-balanced loss, metapath model input, 2.5% ratio.
- ogbn-arxiv: diffusion X0/X1/X2/highpass, block norm, shadow fusion, sqrt-weighted logit-adjusted loss, 6.0% and 12.0% ratios.
- ogbn-products: keep only base shadow-fusion as a completed diagnostic; do not claim streaming diffusion improvement yet.

Do not promote:

- `shadow_fusion_blocknorm` for ACM/DBLP.
- products second-stage streaming diffusion chunk grid.
- any products row whose diffusion blocks were only precomputed but not injected into `phi`.

## Next Recommendation

The next engineering step should be narrow: implement a true lazy feature-block provider so fp16 memmap diffusion blocks can be concatenated into model input without materializing all features in RAM. Only after that should products streaming diffusion be re-tested.
