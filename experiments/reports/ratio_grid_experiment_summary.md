# Ratio Grid Experiment Summary

Generated: 2026-06-07T14:54:22Z

## Scope

- Small grid: `acm`, `dblp`, `imdb`; ratios `0.005, 0.01, 0.02, 0.05`; seeds `0,1,2`; main model `relation_linear`; model compare `relation_mlp` at `0.01,0.02`.
- Medium grid: `ogbn-arxiv`, `ogbn-products`; ratios `0.001, 0.0025, 0.005, 0.01`; model `relation_linear`; loss `sqrt_weighted`; products clipped-loss compare at `0.001,0.0025,0.005`.
- Tables are generated from JSON logs, not manually entered.

## Executive Findings

- Small target-ratio comparison improved materially: Shadow-HGC-R-1 beats the best target-ratio classical baseline on ACM, DBLP, and IMDB.
- IMDB remains the main warning case: Shadow beats target-ratio classical baselines but remains below self-only by more than 3 accuracy points.
- DBLP is the strongest small result: best Shadow accuracy is about `0.8264`, clearly above self-only and classical baselines in this grid.
- ACM best Shadow is `relation_mlp` at ratio `0.02`, about `0.7675`; relation_linear has high variance at ratio `0.05`.
- Medium products improves near-monotonically with ratio under `sqrt_weighted`, and clipped loss is stronger at matched ratios in this run.
- Medium arxiv improves strongly versus previous count-based relation-MLP results, reaching about `0.4294` accuracy, but remains below the earlier full-graph same-backbone sanity result.

## Solidness Gates

| Gate | Status | Reason |
| --- | --- | --- |
| Gate 0: tests | PASS | Full current test suite passed in this run. |
| Gate 1: toy | PASS | Toy main/private/self/full all have 1.0 accuracy and macro-F1. |
| Gate 2: small datasets | WARN | Shadow matches/beats best classical baseline on 3/3 datasets. Self-only gap >3 points on: imdb. |
| Gate 3: medium datasets | WARN | Products is monotonic/near-monotonic; arxiv still needs comparison against self/private/full graph. |
| Gate 4: I/O dry run | PASS | Ratio-aware dry-run logs contain memory/disk/scan fields and cache_all_targets=false. |


## Small Summary

| dataset | best_shadow_model | ratio | acc | acc_std | macro_f1 | best_self_only | best_target_ratio_classical | best_total_node_classical | best_linear_acc | best_mlp_acc | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | relation_mlp | 0.02 | 0.7675 | 0.0056 | 0.7664 | Self-Only-MLP r0.02: 0.7885 | Random-HG r0.05: 0.7025 | Random-HG r0.05: 0.7651 | 0.7672 | 0.7675 | beats target-ratio classical |
| dblp | relation_linear | 0.005 | 0.8264 | 0.0020 | 0.8185 | Self-Only-MLP r0.02: 0.7086 | K-Center-HG r0.05: 0.6535 | K-Center-HG r0.05: 0.6891 | 0.8264 | 0.7202 | beats target-ratio classical |
| imdb | relation_linear | 0.005 | 0.3376 | 0.0059 | 0.2998 | Self-Only-MLP r0.05: 0.4007 | Random-HG r0.05: 0.2917 | K-Center-HG r0.05: 0.3031 | 0.3376 | 0.3098 | beats target-ratio classical; below self-only >3pt |

## Medium Summary

| dataset | loss | model | best_ratio | best_acc | best_macro_f1 | pred_classes | acc_by_ratio | macro_f1_by_ratio | pred_classes_by_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | sqrt_weighted | relation_linear | 0.005 | 0.4294 | 0.2889 | 40 | 0.001:0.3921, 0.0025:0.4288, 0.005:0.4294, 0.01:0.4220 | 0.001:0.2782, 0.0025:0.2927, 0.005:0.2889, 0.01:0.2851 | 0.001:40, 0.0025:40, 0.005:40, 0.01:40 |
| ogbn-products | clipped | relation_linear | 0.005 | 0.4361 | 0.1884 | 33 | 0.001:0.4281, 0.0025:0.4237, 0.005:0.4361 | 0.001:0.1821, 0.0025:0.1846, 0.005:0.1884 | 0.001:35, 0.0025:34, 0.005:33 |
| ogbn-products | sqrt_weighted | relation_linear | 0.005 | 0.4197 | 0.1831 | 41 | 0.001:0.4049, 0.0025:0.4085, 0.005:0.4197, 0.01:0.4171 | 0.001:0.1800, 0.0025:0.1788, 0.005:0.1831, 0.01:0.1852 | 0.001:42, 0.0025:41, 0.005:41, 0.01:41 |

## Small Shadow Rows

| dataset | model | ratio | requested_budget | effective_prototypes | shadow_nodes | condensed_nodes | accuracy | accuracy_std | macro_f1 | shadow_recon_err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | relation_linear | 0.005 | 12 | 12 | 40 | 52 | 0.6014 | 0.1374 | 0.5680 | 0.2856 |
| acm | relation_linear | 0.01 | 12 | 12 | 40 | 52 | 0.6014 | 0.1374 | 0.5680 | 0.2856 |
| acm | relation_linear | 0.02 | 18 | 18 | 40 | 58 | 0.6309 | 0.0765 | 0.5861 | 0.3153 |
| acm | relation_linear | 0.05 | 45 | 45 | 69 | 114 | 0.7672 | 0.1033 | 0.7490 | 0.4076 |
| acm | relation_mlp | 0.01 | 12 | 12 | 40 | 52 | 0.7203 | 0.0329 | 0.7198 | 0.2856 |
| acm | relation_mlp | 0.02 | 18 | 18 | 40 | 58 | 0.7675 | 0.0056 | 0.7664 | 0.3153 |
| dblp | relation_linear | 0.005 | 16 | 16 | 16 | 32 | 0.8264 | 0.0020 | 0.8185 | 0.0237 |
| dblp | relation_linear | 0.01 | 16 | 16 | 16 | 32 | 0.8264 | 0.0020 | 0.8185 | 0.0237 |
| dblp | relation_linear | 0.02 | 24 | 24 | 24 | 48 | 0.8257 | 0.0040 | 0.8179 | 0.0202 |
| dblp | relation_linear | 0.05 | 61 | 61 | 61 | 122 | 0.8261 | 0.0013 | 0.8184 | 0.0616 |
| dblp | relation_mlp | 0.01 | 16 | 16 | 16 | 32 | 0.7202 | 0.0184 | 0.7124 | 0.0237 |
| dblp | relation_mlp | 0.02 | 24 | 24 | 24 | 48 | 0.7175 | 0.0284 | 0.7076 | 0.0202 |
| imdb | relation_linear | 0.005 | 20 | 20 | 24 | 44 | 0.3376 | 0.0059 | 0.2998 | 0.9352 |
| imdb | relation_linear | 0.01 | 20 | 20 | 24 | 44 | 0.3376 | 0.0059 | 0.2998 | 0.9352 |
| imdb | relation_linear | 0.02 | 27 | 27 | 27 | 54 | 0.3057 | 0.0268 | 0.2764 | 0.9637 |
| imdb | relation_linear | 0.05 | 69 | 69 | 69 | 138 | 0.3168 | 0.0183 | 0.2891 | 0.9734 |
| imdb | relation_mlp | 0.01 | 20 | 20 | 24 | 44 | 0.3009 | 0.0248 | 0.2768 | 0.9352 |
| imdb | relation_mlp | 0.02 | 27 | 27 | 27 | 54 | 0.3098 | 0.0254 | 0.2633 | 0.9637 |

## Medium Shadow Rows

| dataset | loss | model | ratio | requested_budget | effective_prototypes | shadow_nodes | condensed_nodes | accuracy | macro_f1 | pred_classes | shadow_recon_err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | sqrt_weighted | relation_linear | 0.001 | 160 | 160 | 80 | 240 | 0.3921 | 0.2782 | 40 | 0.5007 |
| ogbn-arxiv | sqrt_weighted | relation_linear | 0.0025 | 227 | 227 | 114 | 341 | 0.4288 | 0.2927 | 40 | 0.4612 |
| ogbn-arxiv | sqrt_weighted | relation_linear | 0.005 | 455 | 455 | 228 | 683 | 0.4294 | 0.2889 | 40 | 0.4354 |
| ogbn-arxiv | sqrt_weighted | relation_linear | 0.01 | 909 | 909 | 456 | 1365 | 0.4220 | 0.2851 | 40 | 0.4532 |
| ogbn-products | clipped | relation_linear | 0.001 | 197 | 179 | 90 | 269 | 0.4281 | 0.1821 | 35 | 0.4655 |
| ogbn-products | clipped | relation_linear | 0.0025 | 492 | 474 | 238 | 712 | 0.4237 | 0.1846 | 34 | 0.4575 |
| ogbn-products | clipped | relation_linear | 0.005 | 983 | 965 | 484 | 1449 | 0.4361 | 0.1884 | 33 | 0.4406 |
| ogbn-products | sqrt_weighted | relation_linear | 0.001 | 197 | 179 | 90 | 269 | 0.4049 | 0.1800 | 42 | 0.4655 |
| ogbn-products | sqrt_weighted | relation_linear | 0.0025 | 492 | 474 | 238 | 712 | 0.4085 | 0.1788 | 41 | 0.4575 |
| ogbn-products | sqrt_weighted | relation_linear | 0.005 | 983 | 965 | 484 | 1449 | 0.4197 | 0.1831 | 41 | 0.4406 |
| ogbn-products | sqrt_weighted | relation_linear | 0.01 | 1966 | 1948 | 974 | 2922 | 0.4171 | 0.1852 | 41 | 0.4279 |

## Medium Skeleton Ablation

| dataset | setting | accuracy | macro_f1 | pred_classes | skeleton_coverage | residual_energy | shadow_recon_err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | k_s=0 | 0.3561 | 0.2588 | 40 | 0.0000 | 1.0000 | 0.4060 |
| ogbn-arxiv | k_s=1 | 0.3668 | 0.2647 | 40 | 0.3138 | 0.7629 | 0.4339 |
| ogbn-arxiv | k_s=2 | 0.3869 | 0.2727 | 40 | 0.4853 | 0.6618 | 0.4584 |
| ogbn-arxiv | k_s=4 | 0.3921 | 0.2782 | 40 | 0.6452 | 0.5888 | 0.5007 |
| ogbn-arxiv | k_s=8 | 0.3782 | 0.2771 | 40 | 0.7741 | 0.5497 | 0.5188 |
| ogbn-products | k_s=0 | 0.3548 | 0.1554 | 42 | 0.0000 | 1.0000 | 0.4724 |
| ogbn-products | k_s=0 | 0.3798 | 0.1598 | 35 | 0.0000 | 1.0000 | 0.4724 |
| ogbn-products | k_s=1 | 0.3946 | 0.1750 | 42 | 0.5108 | 0.8535 | 0.4638 |
| ogbn-products | k_s=1 | 0.4218 | 0.1825 | 35 | 0.5108 | 0.8535 | 0.4638 |
| ogbn-products | k_s=2 | 0.4012 | 0.1793 | 42 | 0.6380 | 0.7993 | 0.4566 |
| ogbn-products | k_s=2 | 0.4259 | 0.1834 | 34 | 0.6380 | 0.7993 | 0.4566 |
| ogbn-products | k_s=4 | 0.4049 | 0.1800 | 42 | 0.7610 | 0.7798 | 0.4655 |
| ogbn-products | k_s=4 | 0.4281 | 0.1821 | 35 | 0.7610 | 0.7798 | 0.4655 |
| ogbn-products | k_s=8 | 0.4085 | 0.1807 | 42 | 0.8634 | 0.7622 | 0.4842 |
| ogbn-products | k_s=8 | 0.4367 | 0.1845 | 35 | 0.8634 | 0.7622 | 0.4842 |

## Source Artifacts

- `experiments/tables/small_ratio_main.csv`
- `experiments/tables/medium_ratio_main.csv`
- `experiments/tables/ratio_budget_summary.csv`
- `experiments/reports/stage0_4_solidness_report.md`
