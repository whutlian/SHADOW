# Shadow-HGC-R+ Stage Detailed Summary

Generated from R+ stage CSV artifacts in `experiments/tables`.

## Scope

- This report summarizes only the R+ stage requested in `codex_next_stage_rplus_prompt.md`.
- Included tables: `rank_diagnostics_small_medium_seed42.csv`, `imdb_rescue_rplus_seed42.csv`, `medium_diffusion_rplus_seed42.csv`, `acm_dblp_rplus_regression_seed42.csv`.
- Excluded: previous legacy ratio grids, old small/medium ratio sweeps, smoke/current/tmp tables, and pre-R+ summaries.
- Seed policy: single seed 42 for all R+ stage experiments.

## What Changed In This Stage

### New R+ Capabilities

- Added relation-wise rank diagnostics for every train-target relation demand/residual matrix: stable rank, entropy effective rank, reconstruction error, demand norm statistics, and shadow feature norm statistics.
- Added rank-adaptive shadow capacity: relation shadow budgets can be allocated from train-target effective rank without validation/test accuracy.
- Added adaptive nonnegative top-b shadow assignment: b is selected from reconstruction difficulty, edge weights stay nonnegative, and signed shadow features remain allowed.
- Added target multiscale features: diffusion target blocks for medium target-target graphs and schema-preserving meta-path target blocks for small heterogeneous graphs.
- Added coverage-adaptive target-target skeleton: per-row k is chosen to meet retained transition-mass coverage without renormalizing skeleton weights.
- Added optional relation gate in the custom weighted relation-linear layer using positive softplus scalar gates per relation.
- Added `sqrt_weighted_logit_adjusted` prototype loss and prediction-collapse diagnostics: predicted class count, entropy, top classes, per-class support/accuracy, and weighted-F1.

### New Modules And Scripts

- New modules: `shadow_hgc/diagnostics/rank.py`, `shadow_hgc/diagnostics/reconstruction.py`, `shadow_hgc/shadows/adaptive.py`, `shadow_hgc/features/diffusion.py`, `shadow_hgc/features/metapath.py`, `shadow_hgc/features/multiscale.py`, `shadow_hgc/skeleton/policy.py`, `shadow_hgc/eval/class_collapse.py`.
- Updated core modules: `shadow_hgc/pipeline/core.py`, `shadow_hgc/graph/materialize.py`, `shadow_hgc/shadows/assign.py`, `shadow_hgc/skeleton/transition.py`, `shadow_hgc/models/weighted_rel_linear.py`, `shadow_hgc/models/losses.py`, `shadow_hgc/prototype/signatures.py`, `shadow_hgc/config.py`.
- New scripts: `scripts/run_rplus_diagnostics.py`, `scripts/run_imdb_rescue.py`, `scripts/run_medium_diffusion.py`, `scripts/run_rplus_regression.py`.
- Updated general scripts to expose R+ CLI flags: `scripts/run_small.py`, `scripts/run_medium.py`, `scripts/run_medium_ratio_sweep.py`.

### Default-Path Safety

- Shadow-HGC-R-1 defaults remain fixed: `feature_mode=base`, `shadow_policy=fixed`, `adaptive_b=false`, `relation_gate=false`, `skeleton_policy=fixed_k`, and b=1 unless explicitly enabled by R+ flags.
- Exposed graph schema remains original node/edge types only; meta-path features do not create meta-path edge types.
- Edge weights remain nonnegative; destination-row alpha normalization and the custom weighted relation-linear layer remain the message-passing path.

## R+ Artifact Coverage

| Artifact | Rows | Completed | OOM/other | Purpose |
|---|---:|---:|---:|---|
| `rank_diagnostics_small_medium_seed42.csv` | 39 | 39 | 0 | rank/reconstruction hypothesis checks |
| `imdb_rescue_rplus_seed42.csv` | 30 | 30 | 0 | IMDB rescue variants and losses |
| `medium_diffusion_rplus_seed42.csv` | 24 | 16 | 8 | medium diffusion/coverage/logit-adjustment rescue |
| `acm_dblp_rplus_regression_seed42.csv` | 6 | 6 | 0 | ACM/DBLP non-regression checks |

## Rank Diagnostics: All R+ Diagnostic Ratios

The table aggregates relation-level rank rows by dataset and ratio. `Max eff rank relation` identifies the highest-complexity relation at that ratio; `Max recon relation` identifies the hardest reconstructed relation.

| Dataset | Ratio | Relations | Acc | Macro-F1 | Median eff rank | Max eff rank | Max eff rank relation | Median recon | Max recon | Max recon relation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | 0.5% | 5 | 0.5925 | 0.5005 | 1.3756 | 4.9432 | author--writes-->paper | 0.1755 | 0.7883 | author--writes-->paper |
| acm | 2.5% | 5 | 0.6331 | 0.5338 | 1.5669 | 9.2415 | author--writes-->paper | 0.2647 | 0.7781 | author--writes-->paper |
| acm | 9.6% | 5 | 0.6596 | 0.6533 | 1.6898 | 41.4851 | author--writes-->paper | 0.1953 | 0.9093 | author--writes-->paper |
| dblp | 0.5% | 1 | 0.8313 | 0.8237 | 13.0759 | 13.0759 | paper--written_by-->author | 0.0005 | 0.0005 | paper--written_by-->author |
| dblp | 2.5% | 1 | 0.8289 | 0.8213 | 12.5984 | 12.5984 | paper--written_by-->author | 0.0192 | 0.0192 | paper--written_by-->author |
| dblp | 9.6% | 1 | 0.8243 | 0.8165 | 40.2261 | 40.2261 | paper--written_by-->author | 0.0361 | 0.0361 | paper--written_by-->author |
| imdb | 0.5% | 3 | 0.2926 | 0.2849 | 10.2886 | 11.4990 | keyword--keyword_in-->movie | 0.8805 | 0.9378 | director--directs-->movie |
| imdb | 2.5% | 3 | 0.3507 | 0.2956 | 23.6453 | 24.6113 | director--directs-->movie | 0.9895 | 0.9916 | actor--acts_in-->movie |
| imdb | 9.6% | 3 | 0.3004 | 0.2787 | 83.6023 | 85.0120 | director--directs-->movie | 0.9718 | 0.9949 | actor--acts_in-->movie |
| ogbn-arxiv | 0.5% | 2 | 0.4343 | 0.3015 | 19.2412 | 19.2412 | paper--cited_by-->paper | 0.4897 | 0.4897 | paper--cited_by-->paper |
| ogbn-arxiv | 6.0% | 2 | 0.3921 | 0.2564 | 50.9930 | 50.9930 | paper--cited_by-->paper | 0.5978 | 0.5978 | paper--cited_by-->paper |
| ogbn-arxiv | 12.0% | 2 | 0.4664 | 0.3060 | 55.6760 | 55.6760 | paper--cited_by-->paper | 0.5746 | 0.5746 | paper--cited_by-->paper |
| ogbn-products | 0.5% | 2 | 0.4335 | 0.1954 | 32.2697 | 32.2697 | product--co_purchase-->product | 0.4133 | 0.4133 | product--co_purchased_by-->product |
| ogbn-products | 6.0% | 2 | 0.5501 | 0.2451 | 34.7164 | 34.7164 | product--co_purchase-->product | 0.5298 | 0.5298 | product--co_purchase-->product |
| ogbn-products | 12.0% | 2 | 0.5891 | 0.2643 | 35.8950 | 35.8950 | product--co_purchased_by-->product | 0.3716 | 0.3716 | product--co_purchased_by-->product |

## IMDB Rescue: All R+ Ratios, Variants, And Losses

Best base row: `base` / `clipped` at `2.5%`, acc `0.3507`, macro-F1 `0.2956`.
Best full R+ row: `full_rplus` / `clipped` at `0.5%`, acc `0.3810`, macro-F1 `0.3403`.
Best overall IMDB row: `full_rplus` / `clipped` at `0.5%`, acc `0.3810`, macro-F1 `0.3403`.

| Ratio | Variant | Loss | Acc | Macro-F1 | Pred classes | Entropy | Condensed nodes | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5% | base | class_balanced | 0.2826 | 0.2606 | 5 | 1.4761 | 44 | completed |
| 0.5% | base | clipped | 0.2926 | 0.2849 | 5 | 1.5600 | 44 | completed |
| 0.5% | base | sqrt_weighted | 0.3260 | 0.2880 | 5 | 1.4626 | 44 | completed |
| 0.5% | full_rplus | class_balanced | 0.3716 | 0.3370 | 5 | 1.5093 | 126 | completed |
| 0.5% | full_rplus | clipped | 0.3810 | 0.3403 | 5 | 1.4872 | 126 | completed |
| 0.5% | full_rplus | sqrt_weighted | 0.3660 | 0.3335 | 5 | 1.4901 | 126 | completed |
| 2.5% | adaptive_b | class_balanced | 0.3463 | 0.2977 | 5 | 1.4868 | 70 | completed |
| 2.5% | adaptive_b | clipped | 0.3367 | 0.2884 | 5 | 1.4576 | 70 | completed |
| 2.5% | adaptive_b | sqrt_weighted | 0.3154 | 0.2771 | 5 | 1.5317 | 70 | completed |
| 2.5% | base | class_balanced | 0.3307 | 0.2906 | 5 | 1.5383 | 70 | completed |
| 2.5% | base | clipped | 0.3507 | 0.2956 | 5 | 1.4574 | 70 | completed |
| 2.5% | base | sqrt_weighted | 0.3351 | 0.2895 | 5 | 1.4997 | 70 | completed |
| 2.5% | full_rplus | class_balanced | 0.3648 | 0.3207 | 5 | 1.4584 | 238 | completed |
| 2.5% | full_rplus | clipped | 0.3666 | 0.3363 | 5 | 1.5258 | 238 | completed |
| 2.5% | full_rplus | sqrt_weighted | 0.3576 | 0.3370 | 5 | 1.5629 | 238 | completed |
| 2.5% | metapath | class_balanced | 0.3554 | 0.3067 | 5 | 1.5260 | 70 | completed |
| 2.5% | metapath | clipped | 0.3585 | 0.3098 | 5 | 1.5314 | 70 | completed |
| 2.5% | metapath | sqrt_weighted | 0.3513 | 0.3024 | 5 | 1.5310 | 70 | completed |
| 2.5% | rank_adaptive | class_balanced | 0.3517 | 0.3370 | 5 | 1.5859 | 238 | completed |
| 2.5% | rank_adaptive | clipped | 0.3751 | 0.3569 | 5 | 1.5594 | 238 | completed |
| 2.5% | rank_adaptive | sqrt_weighted | 0.3242 | 0.3205 | 5 | 1.5579 | 238 | completed |
| 2.5% | relation_gate | class_balanced | 0.3538 | 0.3100 | 5 | 1.5152 | 70 | completed |
| 2.5% | relation_gate | clipped | 0.3310 | 0.2870 | 5 | 1.5001 | 70 | completed |
| 2.5% | relation_gate | sqrt_weighted | 0.3292 | 0.2846 | 5 | 1.5030 | 70 | completed |
| 5.0% | base | class_balanced | 0.2961 | 0.2788 | 5 | 1.5957 | 138 | completed |
| 5.0% | base | clipped | 0.2945 | 0.2771 | 5 | 1.5863 | 138 | completed |
| 5.0% | base | sqrt_weighted | 0.3079 | 0.2818 | 5 | 1.5576 | 138 | completed |
| 5.0% | full_rplus | class_balanced | 0.3713 | 0.3424 | 5 | 1.5352 | 483 | completed |
| 5.0% | full_rplus | clipped | 0.3485 | 0.3349 | 5 | 1.5699 | 483 | completed |
| 5.0% | full_rplus | sqrt_weighted | 0.3504 | 0.3303 | 5 | 1.5494 | 483 | completed |

### IMDB Interpretation

- Base IMDB rows retain high relation reconstruction errors, especially actor/director/keyword demand in the 2.5% and 5.0% settings.
- Full R+ sharply reduces reconstruction error by allocating more relation shadows from effective-rank diagnostics and adding meta-path target features plus relation gates.
- The best full R+ row improves both accuracy and macro-F1 while keeping all 5 predicted classes, so the gain is not a one-class collapse.

## Medium Diffusion Rescue: All R+ Ratios And Statuses

- ogbn-arxiv: best base `0.4664` at `12.0%`; best included R+ stage row `0.5369` from `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted_logit_adjusted` at `6.0%`.
- ogbn-products: best base `0.5891` at `12.0%`; best included R+ stage row `0.5891` from `base` / `sqrt_weighted` at `12.0%`.

| Dataset | Ratio | Variant | Loss | Acc | Macro-F1 | Pred classes | Skel cov | Mean k | Recon err | Nodes | Edges | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | 0.5% | base | sqrt_weighted | 0.4343 | 0.3015 | 40 | 0.5117 | 3.2582 | 0.4477 | 683 | 3875 | completed |
| ogbn-arxiv | 0.5% | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 0.5189 | 0.3614 | 40 | 0.6080 | 5.4626 | 0.4855 | 683 | 5881 | completed |
| ogbn-arxiv | 2.0% | base | sqrt_weighted | 0.3958 | 0.2668 | 40 | 0.5035 | 3.2188 | 0.5114 | 2729 | 15348 | completed |
| ogbn-arxiv | 2.0% | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 0.4961 | 0.3368 | 40 | 0.6009 | 5.0805 | 0.5454 | 2729 | 22121 | completed |
| ogbn-arxiv | 6.0% | base | sqrt_weighted | 0.3921 | 0.2564 | 40 | 0.6893 | 2.7943 | 0.5849 | 8177 | 41366 | completed |
| ogbn-arxiv | 6.0% | diffusion_X0X1 | sqrt_weighted | 0.4669 | 0.3062 | 40 | 0.7265 | 2.7624 | 0.6140 | 8178 | 41025 | completed |
| ogbn-arxiv | 6.0% | diffusion_X0X1X2 | sqrt_weighted | 0.4922 | 0.3376 | 40 | 0.7311 | 2.7332 | 0.6049 | 8181 | 40714 | completed |
| ogbn-arxiv | 6.0% | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 0.5043 | 0.3318 | 40 | 0.7020 | 3.5609 | 0.6360 | 8172 | 49696 | completed |
| ogbn-arxiv | 6.0% | diffusion_X0X1X2_highpass_coverage | sqrt_weighted_logit_adjusted | 0.5369 | 0.3402 | 39 | 0.7020 | 3.5609 | 0.6360 | 8172 | 49696 | completed |
| ogbn-arxiv | 6.0% | diffusion_highpass_coverage_adaptive | sqrt_weighted | 0.4924 | 0.3273 | 40 | 0.7020 | 3.5609 | 0.7360 | 6199 | 71672 | completed |
| ogbn-arxiv | 12.0% | base | sqrt_weighted | 0.4664 | 0.3060 | 40 | 0.7904 | 2.3804 | 0.5619 | 16251 | 73240 | completed |
| ogbn-arxiv | 12.0% | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 0.5216 | 0.3456 | 40 | 0.7662 | 2.7537 | 0.6099 | 16247 | 81312 | completed |
| ogbn-products | 0.5% | base | sqrt_weighted | 0.4335 | 0.1954 | 41 | 0.6026 | 3.9834 | 0.4129 | 1449 | 9618 | completed |
| ogbn-products | 0.5% | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | NA | NA |  | NA | NA | NA |  |  | oom |
| ogbn-products | 2.0% | base | sqrt_weighted | 0.4697 | 0.2154 | 41 | 0.5233 | 3.9713 | 0.5604 | 5865 | 38866 | completed |
| ogbn-products | 2.0% | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | NA | NA |  | NA | NA | NA |  |  | oom |
| ogbn-products | 6.0% | base | sqrt_weighted | 0.5501 | 0.2451 | 40 | 0.4780 | 3.9539 | 0.5264 | 17454 | 115286 | completed |
| ogbn-products | 6.0% | diffusion_X0X1 | sqrt_weighted | NA | NA |  | NA | NA | NA |  |  | oom |
| ogbn-products | 6.0% | diffusion_X0X1X2 | sqrt_weighted | NA | NA |  | NA | NA | NA |  |  | oom |
| ogbn-products | 6.0% | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | NA | NA |  | NA | NA | NA |  |  | oom |
| ogbn-products | 6.0% | diffusion_X0X1X2_highpass_coverage | sqrt_weighted_logit_adjusted | NA | NA |  | NA | NA | NA |  |  | oom |
| ogbn-products | 6.0% | diffusion_highpass_coverage_adaptive | sqrt_weighted | NA | NA |  | NA | NA | NA |  |  | oom |
| ogbn-products | 12.0% | base | sqrt_weighted | 0.5891 | 0.2643 | 40 | 0.4380 | 3.9457 | 0.3558 | 34174 | 225346 | completed |
| ogbn-products | 12.0% | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | NA | NA |  | NA | NA | NA |  |  | oom |

### Medium OOM / Resource Findings

- ogbn-products `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted` at `0.5%` ended with status `oom`. Source log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p005_seed42.json`
- ogbn-products `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted` at `2.0%` ended with status `oom`. Source log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p02_seed42.json`
- ogbn-products `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted` at `6.0%` ended with status `oom`. Source log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p06_seed42.json`
- ogbn-products `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted` at `12.0%` ended with status `oom`. Source log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p12_seed42.json`
- ogbn-products `diffusion_X0X1` / `sqrt_weighted` at `6.0%` ended with status `oom`. Source log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1_sqrt_weighted_r0p06_seed42.json`
- ogbn-products `diffusion_X0X1X2` / `sqrt_weighted` at `6.0%` ended with status `oom`. Source log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_sqrt_weighted_r0p06_seed42.json`
- ogbn-products `diffusion_highpass_coverage_adaptive` / `sqrt_weighted` at `6.0%` ended with status `oom`. Source log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_highpass_coverage_adaptive_sqrt_weighted_r0p06_seed42.json`
- ogbn-products `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted_logit_adjusted` at `6.0%` ended with status `oom`. Source log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_logit_adjusted_r0p06_seed42.json`

### Medium Interpretation

- ogbn-arxiv benefits from diffusion target features. The best row is diffusion X0/X1/X2/high-pass with coverage skeleton and logit-adjusted loss at 6.0%, reaching 0.5369 accuracy.
- ogbn-products base remains best among completed rows at 12.0%, but every in-memory diffusion row OOMed. This is a resource/scalability result and motivates chunked or memmap diffusion before paper100M.

## ACM/DBLP Regression Checks: All R+ Regression Ratios

| Dataset | Ratio | Variant | Acc | Macro-F1 | Pred classes | Condensed nodes | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acm | 9.6% | base | 0.6596 | 0.6533 | 3 | 218 | completed |
| acm | 9.6% | rplus | 0.8432 | 0.8462 | 3 | 269 | completed |
| dblp | 0.5% | base | 0.8313 | 0.8237 | 4 | 32 | completed |
| dblp | 0.5% | rplus | 0.8282 | 0.8214 | 4 | 48 | completed |
| dblp | 6.5% | base | 0.8197 | 0.8126 | 4 | 158 | completed |
| dblp | 6.5% | rplus | 0.8370 | 0.8299 | 4 | 237 | completed |

### Regression Interpretation

- ACM does not regress under R+ at 9.6%; R+ improves over the base row in this stage.
- DBLP remains stable and strong. The R+ 6.5% row is the best DBLP regression row, but the low-ratio base already performs strongly, matching the rank-saturation hypothesis.

## Stage-Level Conclusions

1. DBLP flatness is explained by low reconstruction error and early relation-demand saturation. Extra ratio is not the main lever for DBLP.
2. IMDB failure is explained by high-rank, hard-to-reconstruct non-target demand. R+ improves IMDB by increasing relation shadow capacity and adding schema-preserving meta-path target features.
3. ogbn-arxiv benefits from diffusion/high-pass target features and coverage skeleton. The best R+ row is materially above the base row from the same stage.
4. ogbn-products cannot use the current in-memory diffusion path safely; the correct next engineering step is chunked/memmap diffusion rather than more ratio sweeps.
5. R+ remains inside the Shadow-HGC relation-demand condensation framework: no dense synthetic adjacency learning, no new exposed meta-path edge types, no negative edge weights, and no validation/test-driven shadow allocation.

## Verification

- Full test suite after R+ implementation: `81 passed in 80.93s` using `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests -q`.
- R+ deliverable checklist: 33 required files present and non-empty.

## R+ Source Files

- Tables: `experiments/tables/rank_diagnostics_small_medium_seed42.csv`, `experiments/tables/imdb_rescue_rplus_seed42.csv`, `experiments/tables/medium_diffusion_rplus_seed42.csv`, `experiments/tables/acm_dblp_rplus_regression_seed42.csv`.
- Reports: `experiments/reports/rank_diagnostics_summary.md`, `experiments/reports/imdb_rescue_rplus_summary.md`, `experiments/reports/medium_diffusion_rplus_summary.md`, `experiments/reports/rplus_rescue_summary.md`, `experiments/reports/stage5_readiness_after_rplus.md`.
- Stage-5 selected config: `configs/stage5_selected.yaml`.
