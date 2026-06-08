# Three Experiment Result Summary

## Scope

- Experiment 1: DBLP / ACM / IMDB at requested ratios 1.2%, 2.4%, 4.8%, 9.6%; table values are mean/std over seeds in the source logs.
- Experiment 2: DBLP / IMDB fine ratio sweep from 0.5% to 12.0%, single seed 0.
- Experiment 3: ogbn-arxiv / ogbn-products medium fine ratio sweep from 0.5% to 12.0%, single seed 42.
- Ratio is requested target prototype ratio. Condensed node count is larger because shadow nodes are included.

## Executive Takeaways

### Experiment 1: Fixed Small Ratios

| Dataset | Best ratio by accuracy | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std | Condensed nodes (seed0) |
|---|---:|---:|---:|---:|---:|---:|
| DBLP | 2.4% | 0.8276 | 0.0007 | 0.8202 | 0.0010 | 58 |
| ACM | 9.6% | 0.8573 | 0.0089 | 0.8577 | 0.0093 | 218 |
| IMDB | 1.2% | 0.3376 | 0.0059 | 0.2998 | 0.0063 | 44 |

| Dataset | 1.2% acc | 2.4% acc | 4.8% acc | 9.6% acc | Direction |
|---|---:|---:|---:|---:|---|
| DBLP | 0.8264 | 0.8276 | 0.8269 | 0.8268 | mostly flat / noisy |
| ACM | 0.6014 | 0.6829 | 0.7137 | 0.8573 | clear gain with larger ratio |
| IMDB | 0.3376 | 0.3181 | 0.3142 | 0.2967 | drops at larger ratio |

### Experiment 2: DBLP / IMDB Fine Sweep

| Dataset | Best ratio by accuracy | Accuracy | Macro-F1 at best acc | Best ratio by macro-F1 | Best macro-F1 | Accuracy at best F1 | Condensed nodes at best acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| DBLP | 6.5% | 0.8331 | 0.8254 | 6.5% | 0.8254 | 0.8331 | 158 |
| IMDB | 0.5% | 0.3376 | 0.2932 | 2.5% | 0.3074 | 0.3264 | 44 |

| Dataset | Low-ratio acc (0.5%) | High-ratio acc (12.0%) | Range observation |
|---|---:|---:|---|
| DBLP | 0.8289 | 0.8282 | broad plateau; best is mid-range, not the largest ratio |
| IMDB | 0.3376 | 0.2767 | accuracy is best at very low ratio; macro-F1 peaks around a small ratio |

### Experiment 3: Medium Fine Sweep

| Dataset | Best ratio by accuracy | Accuracy | Macro-F1 at best acc | Best ratio by macro-F1 | Best macro-F1 | Condensed nodes at best acc |
|---|---:|---:|---:|---:|---:|---:|
| ogbn-arxiv | 12.0% | 0.4697 | 0.3103 | 12.0% | 0.3103 | 16251 |
| ogbn-products | 12.0% | 0.5891 | 0.2643 | 12.0% | 0.2643 | 34174 |

| Dataset | 0.5% acc | 6.0% acc | 9.0% acc | 12.0% acc | Trend |
|---|---:|---:|---:|---:|---|
| ogbn-arxiv | 0.4293 | 0.3916 | 0.4290 | 0.4697 | recovers after mid-ratio dip; best at 12% |
| ogbn-products | 0.4295 | 0.5476 | 0.5753 | 0.5891 | mostly monotonic upward; best at 12% |

## Interpretation

- ACM in the fixed-ratio experiment shows the clearest expected gain from larger requested target budget: mean accuracy rises from 0.6014 at 1.2% to 0.8573 at 9.6%.
- DBLP is already strong at very low budget. The fixed-ratio result is almost flat, while the fine single-seed sweep peaks at 6.5%, suggesting extra budget helps only mildly and non-monotonically.
- IMDB remains weak and noisy. Larger ratios do not reliably improve accuracy; the best fine-sweep accuracy is at the smallest tested ratio, while macro-F1 peaks around 2.5%.
- Medium datasets benefit more from larger budgets under the current relation-linear setting. Both ogbn-arxiv and ogbn-products peak at 12.0%, with products showing the cleanest upward trend.
- The medium sweep required two implementation fixes to complete safely: sparse target-target top-k skeleton computation and no-grad original-graph inference. After fixes, the full test suite passed.

## Ratio Grid Results Included

This section folds in all grid CSVs under `experiments/tables` whose filenames contain `grid`. The older ratio-grid analysis is preserved here as historical context. Note that these grids use earlier ratio ranges than the three new experiments above: small legacy grid ratios were `0.5%, 1%, 2%, 5%`; medium legacy grid ratios were `0.1%, 0.25%, 0.5%, 1%`. Therefore the grid best points are not directly interchangeable with the later fixed-ratio and fine-sweep results.

### Grid Table Coverage

| Grid table | Rows | Completed | Other statuses | Datasets | Methods / ablations |
|---|---:|---:|---|---|---|
| `small_ratio_grid_legacy.csv` | 87 | 87 | - | acm, dblp, imdb | 6 methods |
| `small_ratio_grid_ablation.csv` | 40 | 38 | 2 not_applicable | acm, dblp, imdb | 8 ablations |
| `medium_ratio_grid_legacy.csv` | 43 | 43 | - | ogbn-arxiv, ogbn-products | 5 methods |
| `medium_ratio_grid_ablation.csv` | 10 | 10 | - | ogbn-arxiv, ogbn-products | 1 ablation |

### Legacy Grid Best By Dataset / Method

Small-grid values use `accuracy_mean` / `macro_f1_mean` over seeds. `M_tau` is requested target prototype count in the legacy raw table, not a percent.

| Dataset | Method | Best acc | Acc std | Macro-F1 | Macro-F1 std | M_tau | Model / loss |
|---|---|---:|---:|---:|---:|---:|---|
| acm | Full-WRL-GNN | 0.8015 | 0.0056 | 0.8058 | 0.0051 | - | relation_linear / clipped |
| acm | Herding-HG | 0.7486 | 0.0111 | - | - | 114 | - |
| acm | K-Center-HG | 0.7559 | 0.0099 | - | - | 114 | - |
| acm | Random-HG | 0.7651 | 0.0215 | - | - | 114 | - |
| acm | Self-Only-MLP | 0.7885 | 0.0474 | 0.7832 | 0.0544 | 18 | relation_mlp / clipped |
| acm | Shadow-HGC-R-1 | 0.7675 | 0.0056 | 0.7664 | 0.0072 | 18 | relation_mlp / clipped |
| dblp | Full-WRL-GNN | 0.8107 | 0.0014 | 0.8021 | 0.0015 | - | relation_linear / clipped |
| dblp | Herding-HG | 0.6038 | 0.0038 | - | - | 122 | - |
| dblp | K-Center-HG | 0.6891 | 0.0166 | - | - | 122 | - |
| dblp | Random-HG | 0.6038 | 0.0155 | - | - | 122 | - |
| dblp | Self-Only-MLP | 0.7086 | 0.0079 | 0.6972 | 0.0105 | 24 | relation_mlp / clipped |
| dblp | Shadow-HGC-R-1 | 0.8264 | 0.0020 | 0.8185 | 0.0020 | 16 | relation_linear / clipped |
| imdb | Full-WRL-GNN | 0.3351 | 0.0030 | 0.3086 | 0.0030 | - | relation_linear / clipped |
| imdb | Herding-HG | 0.2565 | 0.0006 | - | - | 138 | - |
| imdb | K-Center-HG | 0.3031 | 0.0111 | - | - | 138 | - |
| imdb | Random-HG | 0.2944 | 0.0116 | - | - | 54 | - |
| imdb | Self-Only-MLP | 0.4007 | 0.0082 | 0.3596 | 0.0088 | 69 | relation_mlp / clipped |
| imdb | Shadow-HGC-R-1 | 0.3376 | 0.0059 | 0.2998 | 0.0063 | 20 | relation_linear / clipped |

Medium-grid values are single-seed grid rows.

| Dataset | Method | Best acc | Macro-F1 | M_tau | Model / loss | Shadow mode | Self only |
|---|---|---:|---:|---:|---|---|---|
| ogbn-arxiv | Herding-HG | 0.3269 | - | 227 | - | - | - |
| ogbn-arxiv | K-Center-HG | 0.2196 | - | 909 | - | - | - |
| ogbn-arxiv | Random-HG | 0.2993 | - | 909 | - | - | - |
| ogbn-arxiv | Self-Only-MLP | 0.3101 | 0.1682 | 909 | relation_mlp / sqrt_weighted | virtual_demand_shadow | true |
| ogbn-arxiv | Shadow-HGC-R-1 | 0.4294 | 0.2889 | 455 | relation_linear / sqrt_weighted | virtual_demand_shadow | false |
| ogbn-products | Herding-HG | 0.2715 | - | 179 | - | - | - |
| ogbn-products | K-Center-HG | 0.1774 | - | 1948 | - | - | - |
| ogbn-products | Random-HG | 0.3641 | - | 1948 | - | - | - |
| ogbn-products | Self-Only-MLP | 0.4010 | 0.1395 | 1966 | relation_mlp / sqrt_weighted | virtual_demand_shadow | true |
| ogbn-products | Shadow-HGC-R-1 | 0.4361 | 0.1884 | 983 | relation_linear / clipped | virtual_demand_shadow | false |

### Grid Ablation Best Points

Small-grid ablations report the best setting per dataset and ablation family.

| Dataset | Ablation | Best setting | Acc | Macro-F1 | Skeleton coverage | Residual energy | Shadow recon err |
|---|---|---|---:|---:|---:|---:|---:|
| acm | backbone | model=relation_mlp | 0.7653 | 0.7641 | 0.3425 | 1.0000 | 0.3068 |
| acm | mean_only_demand | include_degree_features=false | 0.6534 | 0.5687 | 0.3650 | 1.0000 | 0.1845 |
| acm | private_shadow_upper_bound | shadow_mode=private_shadow | 0.8650 | 0.8665 | 0.3425 | 1.0000 | 0.0000 |
| acm | prototype_loss | loss_type=sqrt_weighted | 0.8565 | 0.8565 | 0.3425 | 1.0000 | 0.3068 |
| acm | real_source_centroid | shadow_mode=real_source_centroid | 0.6478 | 0.6472 | 0.3425 | 1.0000 | 0.8213 |
| acm | relation_norm_calibration | calibration_enabled=false | 0.7384 | 0.7343 | 0.3425 | 1.0000 | 0.2569 |
| acm | residual_shadow_off | residual_shadow=false | 0.8517 | 0.8476 | 0.3425 | 1.0000 | 0.5692 |
| acm | target_target_skeleton | k_s=1 | 0.7767 | 0.7727 | 0.2342 | 1.0000 | 0.3068 |
| dblp | backbone | model=relation_linear | 0.8289 | 0.8209 | 0.0000 | 1.0000 | 0.0132 |
| dblp | mean_only_demand | include_degree_features=false | 0.8310 | 0.8232 | 0.0000 | 1.0000 | 0.0003 |
| dblp | private_shadow_upper_bound | shadow_mode=private_shadow | 0.8285 | 0.8206 | 0.0000 | 1.0000 | 0.0000 |
| dblp | prototype_loss | loss_type=clipped | 0.8289 | 0.8209 | 0.0000 | 1.0000 | 0.0132 |
| dblp | real_source_centroid | shadow_mode=real_source_centroid | 0.7665 | 0.7607 | 0.0000 | 1.0000 | 1.0076 |
| dblp | relation_norm_calibration | calibration_enabled=false | 0.8289 | 0.8209 | 0.0000 | 1.0000 | 0.0132 |
| dblp | residual_shadow_off | residual_shadow=false | 0.8289 | 0.8209 | 0.0000 | 1.0000 | 0.0132 |
| imdb | backbone | model=relation_linear | 0.3376 | 0.2932 | 0.0000 | 1.0000 | 0.9053 |
| imdb | mean_only_demand | include_degree_features=false | 0.3360 | 0.2949 | 0.0000 | 1.0000 | 0.9770 |
| imdb | private_shadow_upper_bound | shadow_mode=private_shadow | 0.3757 | 0.3561 | 0.0000 | 1.0000 | 0.0000 |
| imdb | prototype_loss | loss_type=class_balanced | 0.3513 | 0.3209 | 0.0000 | 1.0000 | 0.9053 |
| imdb | real_source_centroid | shadow_mode=real_source_centroid | 0.2854 | 0.2634 | 0.0000 | 1.0000 | 1.0413 |
| imdb | relation_norm_calibration | calibration_enabled=false | 0.3370 | 0.2897 | 0.0000 | 1.0000 | 0.8240 |
| imdb | residual_shadow_off | residual_shadow=false | 0.3376 | 0.2932 | 0.0000 | 1.0000 | 0.9053 |

Medium-grid skeleton ablation shows the top-k residual skeleton trend.

| Dataset | k_s | Acc | Macro-F1 | Skeleton coverage | Residual energy | Shadow recon err |
|---|---:|---:|---:|---:|---:|---:|
| ogbn-arxiv | 0 | 0.3561 | 0.2588 | 0.0000 | 1.0000 | 0.4060 |
| ogbn-arxiv | 1 | 0.3668 | 0.2647 | 0.3138 | 0.7629 | 0.4339 |
| ogbn-arxiv | 2 | 0.3869 | 0.2727 | 0.4853 | 0.6618 | 0.4584 |
| ogbn-arxiv | 4 | 0.3921 | 0.2782 | 0.6452 | 0.5888 | 0.5007 |
| ogbn-arxiv | 8 | 0.3782 | 0.2771 | 0.7741 | 0.5497 | 0.5188 |
| ogbn-products | 0 | 0.3548 | 0.1554 | 0.0000 | 1.0000 | 0.4724 |
| ogbn-products | 1 | 0.3946 | 0.1750 | 0.5108 | 0.8535 | 0.4638 |
| ogbn-products | 2 | 0.4012 | 0.1793 | 0.6380 | 0.7993 | 0.4566 |
| ogbn-products | 4 | 0.4049 | 0.1800 | 0.7610 | 0.7798 | 0.4655 |
| ogbn-products | 8 | 0.4085 | 0.1807 | 0.8634 | 0.7622 | 0.4842 |

### Preserved Grid Analysis

- Small target-ratio comparison improved materially: Shadow-HGC-R-1 beats the best target-ratio classical baseline on ACM, DBLP, and IMDB in the legacy grid.
- IMDB remains the warning case: Shadow beats target-ratio classical baselines but remains below self-only by more than 3 accuracy points in the legacy grid.
- DBLP is the strongest small grid result: best Shadow accuracy is `0.8264`, clearly above self-only and classical baselines in that grid.
- ACM's best Shadow row in the legacy grid is `relation_mlp` at ratio `2%`, accuracy `0.7675`; the later fixed-ratio experiment reaches `0.8573` at requested ratio `9.6%`, so the newer ACM result supersedes the old grid ceiling.
- Medium products improves with ratio in the later fine sweep, while the legacy medium grid shows clipped loss stronger than sqrt-weighted at matched low ratios.
- Medium arxiv reaches `0.4294` in the legacy low-ratio grid and `0.4697` in the later 12% fine sweep; the newer result is the current best among the included medium experiments.

## Source Files

- Fixed small ratios: `experiments\tables\small_requested_ratio_accuracy_20260608.csv`
- DBLP/IMDB fine sweep: `experiments\tables\dblp_imdb_ratio_sweep_seed0_20260608.csv`
- Medium fine sweep: `experiments\tables\medium_ratio_sweep_seed42_20260608.csv`
- Small legacy grid: `experiments\tables\small_ratio_grid_legacy.csv`
- Small ablation grid: `experiments\tables\small_ratio_grid_ablation.csv`
- Medium legacy grid: `experiments\tables\medium_ratio_grid_legacy.csv`
- Medium ablation grid: `experiments\tables\medium_ratio_grid_ablation.csv`
- Historical grid report retained for detailed row-level context: `experiments\reports\ratio_grid_experiment_summary.md`
- Supplementary grid figure data: `experiments\figures\small_ratio_grid.csv`, `experiments\figures\small_ratio_grid.svg`
