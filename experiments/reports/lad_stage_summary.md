# Shadow-HGC-L / LAD Stage Summary

## 1. Scope

- Seed policy: single seed `42` only.
- No-diffusion decision: diffusion is not promoted because it caused OOM/resource failures on products and is too expensive for large-scale goals. It remains an appendix diagnostic only.
- Variants: V0 current_best, V1 compiled demand head, V2 compiled demand head + LAD, V3 V2 + boundary-aware prototypes.
- Ratios: ACM 9.6%; DBLP 0.5% and 6.5%; IMDB 0.5%, 2.5%, 5.0%; ogbn-arxiv/products 6.0% and 12.0%.

## 2. Code Changes

- Added train-label-only LAD feature computation in `shadow_hgc/features/label_affinity.py`.
- Added compiled demand schema/table helpers in `shadow_hgc/features/compiled_table.py`.
- Added block-gated compiled demand MLP in `shadow_hgc/models/compiled_demand.py`.
- Added boundary-aware prototype helper in `shadow_hgc/prototype/boundary.py`.
- Integrated opt-in LAD/compiled/boundary arguments into `shadow_hgc/pipeline/core.py` without changing the default R-1 path.
- Added LAD scripts under `scripts/run_lad_*.py` and tests under `tests/test_*lad*`, `tests/test_compiled_*`, and `tests/test_boundary_*`.
- Test command: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests -q`.
- Latest result after experiments: `112 passed in 63.63s`.

Diffusion status:

- Diffusion is not promoted because it caused OOM/resource failures on products and is too expensive for large-scale goals. It remains an appendix diagnostic only.

## 3. Main Results

- acm: `V0_current_best` at ratio `0.0960` reached accuracy `0.8173` / macro-F1 `0.8177`.
- dblp: `V0_current_best` at ratio `0.0650` reached accuracy `0.8370` / macro-F1 `0.8300`.
- imdb: `V2_compiled_plus_lad` at ratio `0.0050` reached accuracy `0.4200` / macro-F1 `0.3552`.
- ogbn-arxiv: `V2_compiled_plus_lad` at ratio `0.1200` reached accuracy `0.5968` / macro-F1 `0.4155`.
- ogbn-products: `V2_compiled_plus_lad` at ratio `0.1200` reached accuracy `0.6587` / macro-F1 `0.3381`.

### Small Datasets

| Dataset | Ratio | Variant | Status | Acc | Macro-F1 | Pred classes | Compiled | LAD | Boundary |
|---|---:|---|---|---:|---:|---:|---|---|---|
| acm | 0.0960 | V0_current_best | completed | 0.8173 | 0.8177 | 3 | False | False | False |
| acm | 0.0960 | V1_compiled_demand_head | completed | 0.7960 | 0.7926 | 3 | True | False | False |
| acm | 0.0960 | V2_compiled_plus_lad | completed | 0.8083 | 0.8105 | 3 | True | True | False |
| acm | 0.0960 | V3_compiled_lad_boundary | completed | 0.8050 | 0.8073 | 3 | True | True | True |
| dblp | 0.0050 | V0_current_best | completed | 0.8268 | 0.8201 | 4 | False | False | False |
| dblp | 0.0050 | V1_compiled_demand_head | completed | 0.3965 | 0.3862 | 4 | True | False | False |
| dblp | 0.0050 | V2_compiled_plus_lad | completed | 0.4320 | 0.4310 | 4 | True | True | False |
| dblp | 0.0050 | V3_compiled_lad_boundary | completed | 0.2507 | 0.1002 | 1 | True | True | True |
| dblp | 0.0650 | V0_current_best | completed | 0.8370 | 0.8300 | 4 | False | False | False |
| dblp | 0.0650 | V1_compiled_demand_head | completed | 0.4754 | 0.4922 | 4 | True | False | False |
| dblp | 0.0650 | V2_compiled_plus_lad | completed | 0.4183 | 0.4266 | 4 | True | True | False |
| dblp | 0.0650 | V3_compiled_lad_boundary | completed | 0.4542 | 0.4680 | 4 | True | True | True |
| imdb | 0.0050 | V0_current_best | completed | 0.3339 | 0.3256 | 5 | False | False | False |
| imdb | 0.0050 | V1_compiled_demand_head | completed | 0.3823 | 0.3541 | 5 | True | False | False |
| imdb | 0.0050 | V2_compiled_plus_lad | completed | 0.4200 | 0.3552 | 5 | True | True | False |
| imdb | 0.0050 | V3_compiled_lad_boundary | completed | 0.3523 | 0.3478 | 5 | True | True | True |
| imdb | 0.0250 | V0_current_best | completed | 0.4076 | 0.3842 | 5 | False | False | False |
| imdb | 0.0250 | V1_compiled_demand_head | completed | 0.3707 | 0.3286 | 5 | True | False | False |
| imdb | 0.0250 | V2_compiled_plus_lad | completed | 0.3879 | 0.3608 | 5 | True | True | False |
| imdb | 0.0250 | V3_compiled_lad_boundary | completed | 0.3819 | 0.3481 | 5 | True | True | True |
| imdb | 0.0500 | V0_current_best | completed | 0.3351 | 0.3297 | 5 | False | False | False |
| imdb | 0.0500 | V1_compiled_demand_head | completed | 0.3888 | 0.3627 | 5 | True | False | False |
| imdb | 0.0500 | V2_compiled_plus_lad | completed | 0.3791 | 0.3549 | 5 | True | True | False |
| imdb | 0.0500 | V3_compiled_lad_boundary | completed | 0.3645 | 0.3398 | 5 | True | True | True |

### Medium Datasets

| Dataset | Ratio | Variant | Status | Acc | Macro-F1 | Pred classes | Compiled | LAD | Boundary |
|---|---:|---|---|---:|---:|---:|---|---|---|
| ogbn-arxiv | 0.0600 | V0_current_best | completed | 0.4999 | 0.3040 | 40 | False | False | False |
| ogbn-arxiv | 0.0600 | V1_compiled_demand_head | completed | 0.4818 | 0.2816 | 39 | True | False | False |
| ogbn-arxiv | 0.0600 | V2_compiled_plus_lad | completed | 0.5902 | 0.4077 | 39 | True | True | False |
| ogbn-arxiv | 0.0600 | V3_compiled_lad_boundary | completed | 0.5765 | 0.4148 | 40 | True | True | True |
| ogbn-arxiv | 0.1200 | V0_current_best | completed | 0.5289 | 0.3422 | 40 | False | False | False |
| ogbn-arxiv | 0.1200 | V1_compiled_demand_head | completed | 0.5200 | 0.3244 | 38 | True | False | False |
| ogbn-arxiv | 0.1200 | V2_compiled_plus_lad | completed | 0.5968 | 0.4155 | 40 | True | True | False |
| ogbn-arxiv | 0.1200 | V3_compiled_lad_boundary | completed | 0.5948 | 0.3737 | 39 | True | True | True |
| ogbn-products | 0.0600 | V0_current_best | completed | 0.5909 | 0.2281 | 35 | False | False | False |
| ogbn-products | 0.0600 | V1_compiled_demand_head | completed | 0.4446 | 0.1846 | 29 | True | False | False |
| ogbn-products | 0.0600 | V2_compiled_plus_lad | completed | 0.6223 | 0.3307 | 32 | True | True | False |
| ogbn-products | 0.0600 | V3_compiled_lad_boundary | completed | 0.6043 | 0.3098 | 33 | True | True | True |
| ogbn-products | 0.1200 | V0_current_best | completed | 0.6081 | 0.2368 | 33 | False | False | False |
| ogbn-products | 0.1200 | V1_compiled_demand_head | completed | 0.4658 | 0.1974 | 29 | True | False | False |
| ogbn-products | 0.1200 | V2_compiled_plus_lad | completed | 0.6587 | 0.3381 | 31 | True | True | False |
| ogbn-products | 0.1200 | V3_compiled_lad_boundary | completed | 0.6146 | 0.3171 | 32 | True | True | True |

## 4. Diagnostic Results

### Upper-Bound Diagnostics

| Dataset | Ratio | Variant | Status | Acc | Macro-F1 | Pred classes | Compiled | LAD | Boundary |
|---|---:|---|---|---:|---:|---:|---|---|---|
| imdb | 0.0250 | FullDemandTable-MLP | completed | 0.3798 | 0.3427 | 5 | True | True | False |
| ogbn-arxiv | 0.1200 | FullDemandTable-MLP | completed | 0.6616 | 0.4025 | 36 | True | True | False |
| ogbn-products | 0.1200 | FullDemandTable-MLP | completed | 0.6884 | 0.3391 | 28 | True | True | False |
| acm | 0.0960 | PrototypeOracleDemand-MLP | completed | 0.8423 | 0.8380 | 3 | True | True | False |
| imdb | 0.0250 | PrototypeOracleDemand-MLP | completed | 0.4460 | 0.3852 | 5 | True | True | False |
| ogbn-arxiv | 0.1200 | PrototypeOracleDemand-MLP | completed | 0.6143 | 0.4351 | 40 | True | True | False |
| ogbn-products | 0.1200 | PrototypeOracleDemand-MLP | completed | 0.6576 | 0.3470 | 32 | True | True | False |

Completed diagnostic rows: 7 / 7.

Diagnostic interpretation:

- IMDB: `PrototypeOracleDemand-MLP` reaches `0.4460` accuracy, while the best condensed LAD row is `0.4200` at 0.5% and `0.3879` at 2.5%. The gap indicates shadow/prototype reconstruction is a bottleneck. `FullDemandTable-MLP` is only `0.3798`, so the full train-table head is also not strong enough in this configuration.
- ogbn-arxiv: `FullDemandTable-MLP` reaches `0.6616`, while best condensed LAD reaches `0.5968`. This indicates remaining condensation/prototype loss. `PrototypeOracleDemand-MLP` reaches `0.6143`, so shadow reconstruction costs about `0.0175` accuracy at 12%.
- ogbn-products: `FullDemandTable-MLP` reaches `0.6884`, still below the 0.70 target but above condensed LAD. `PrototypeOracleDemand-MLP` is `0.6576`, almost identical to V2 `0.6587`, so shadow factorization is not the main products bottleneck; the signal/head ceiling is.

## 5. LAD Analysis

- LAD uses training labels only; validation/test labels are not used in label-affinity construction.
- LAD blocks are target-side compiled features, not exposed graph edge types.
- LAD block statistics and learned block gates are logged in per-run JSON files.
- Medium LAD gains are strong: at 12%, ogbn-arxiv V2 improves over V1 from `0.5200` to `0.5968`; ogbn-products V2 improves from `0.4658` to `0.6587`.
- Small LAD is mixed: ACM V2 improves over V1 but remains below the R++ ACM best; DBLP compiled rows are much worse than V0; IMDB improves at 0.5% but does not beat the 2.5% R++ best.

## 6. Boundary Prototype Analysis

- V3 enables boundary-aware prototypes with `boundary_fraction=0.30` and train-only entropy scoring.
- Boundary pool sizes, score stats, and base/boundary prototype counts are logged in V3 JSON files.
- Boundary prototypes are not promoted from this stage. V3 generally underperforms V2, except a small macro-F1 improvement on ogbn-arxiv 6%; it hurts products and IMDB.

## 7. Compression and Resource Accounting

- Tables include target ratio, total condensed node ratio, byte-size compression, LAD precompute time, CPU RAM, and GPU RAM fields.
- FullDemandTable diagnostics are upper bounds and should not be read as condensation compression results.
- `Full-graph condensed node ratio` is `condensed_nodes_total / original_nodes_total`, so it can differ sharply from the requested target prototype ratio.

### Small Full-Graph Condensed Node Ratio

| Dataset | Requested target ratio | Variant | Full-graph condensed node ratio | Condensed nodes |
|---|---:|---|---:|---:|
| acm | 9.6% | `V0_current_best` | 2.458% | 269 |
| acm | 9.6% | `V1_compiled_demand_head` | 2.458% | 269 |
| acm | 9.6% | `V2_compiled_plus_lad` | 2.458% | 269 |
| acm | 9.6% | `V3_compiled_lad_boundary` | 2.248% | 246 |
| dblp | 0.5% | `V0_current_best` | 0.184% | 48 |
| dblp | 0.5% | `V1_compiled_demand_head` | 0.184% | 48 |
| dblp | 0.5% | `V2_compiled_plus_lad` | 0.184% | 48 |
| dblp | 0.5% | `V3_compiled_lad_boundary` | 0.092% | 24 |
| dblp | 6.5% | `V0_current_best` | 0.907% | 237 |
| dblp | 6.5% | `V1_compiled_demand_head` | 0.907% | 237 |
| dblp | 6.5% | `V2_compiled_plus_lad` | 0.907% | 237 |
| dblp | 6.5% | `V3_compiled_lad_boundary` | 0.700% | 183 |
| imdb | 0.5% | `V0_current_best` | 0.593% | 127 |
| imdb | 0.5% | `V1_compiled_demand_head` | 0.593% | 127 |
| imdb | 0.5% | `V2_compiled_plus_lad` | 0.593% | 127 |
| imdb | 0.5% | `V3_compiled_lad_boundary` | 0.621% | 133 |
| imdb | 2.5% | `V0_current_best` | 1.106% | 237 |
| imdb | 2.5% | `V1_compiled_demand_head` | 1.106% | 237 |
| imdb | 2.5% | `V2_compiled_plus_lad` | 1.106% | 237 |
| imdb | 2.5% | `V3_compiled_lad_boundary` | 1.111% | 238 |
| imdb | 5.0% | `V0_current_best` | 2.255% | 483 |
| imdb | 5.0% | `V1_compiled_demand_head` | 2.255% | 483 |
| imdb | 5.0% | `V2_compiled_plus_lad` | 2.255% | 483 |
| imdb | 5.0% | `V3_compiled_lad_boundary` | 2.255% | 483 |

### Medium Full-Graph Condensed Node Ratio

| Dataset | Requested target ratio | Variant | Full-graph condensed node ratio | Condensed nodes |
|---|---:|---|---:|---:|
| ogbn-arxiv | 6.0% | `V0_current_best` | 3.664% | 6204 |
| ogbn-arxiv | 6.0% | `V1_compiled_demand_head` | 3.664% | 6204 |
| ogbn-arxiv | 6.0% | `V2_compiled_plus_lad` | 3.664% | 6204 |
| ogbn-arxiv | 6.0% | `V3_compiled_lad_boundary` | 3.603% | 6101 |
| ogbn-arxiv | 12.0% | `V0_current_best` | 6.942% | 11755 |
| ogbn-arxiv | 12.0% | `V1_compiled_demand_head` | 6.942% | 11755 |
| ogbn-arxiv | 12.0% | `V2_compiled_plus_lad` | 6.942% | 11755 |
| ogbn-arxiv | 12.0% | `V3_compiled_lad_boundary` | 6.951% | 11771 |
| ogbn-products | 6.0% | `V0_current_best` | 0.502% | 12300 |
| ogbn-products | 6.0% | `V1_compiled_demand_head` | 0.502% | 12300 |
| ogbn-products | 6.0% | `V2_compiled_plus_lad` | 0.502% | 12300 |
| ogbn-products | 6.0% | `V3_compiled_lad_boundary` | 0.508% | 12453 |
| ogbn-products | 12.0% | `V0_current_best` | 0.960% | 23500 |
| ogbn-products | 12.0% | `V1_compiled_demand_head` | 0.960% | 23500 |
| ogbn-products | 12.0% | `V2_compiled_plus_lad` | 0.960% | 23501 |
| ogbn-products | 12.0% | `V3_compiled_lad_boundary` | 0.978% | 23960 |

## 8. Decision

- Promote LAD for medium no-diffusion experiments and as the replacement signal to study instead of diffusion. Do not promote LAD as a universal small-dataset default.
- Do not promote compiled head alone. V1 usually underperforms V0; its value appears only when paired with LAD on medium datasets.
- Do not promote boundary prototypes as a default. Keep V3 as an ablation until the scoring/allocation strategy is improved.
- Return to large-scale stage only after improving the compiled/prototype path for products, because products accuracy is `0.6587`, below the `0.70` target, despite macro-F1 improving beyond the R++ `0.308` reference.

Direct bottleneck answers:

- Is the bottleneck condensation? Partly yes for arxiv/products: FullDemandTable is higher than condensed V2. For IMDB, the stronger oracle prototype row points more specifically to prototype/shadow reconstruction.
- Is the bottleneck shadow factorization? Yes for IMDB and mildly for arxiv; no for products, where PrototypeOracle and V2 are effectively tied.
- Is the bottleneck training head? Compiled head alone is not enough. V1 is weak, but FullDemandTable shows the same head can be strong when trained on all target rows for arxiv/products, so the head is not the only bottleneck.
- Is LAD useful enough to replace diffusion? For arxiv, yes as a scalable no-diffusion promoted path because V2 reaches `0.5968` and passes the `0.58` gate. For products, LAD improves macro-F1 to `0.3381` but does not reach `0.70` accuracy, so it is promising but not sufficient. For small datasets, no universal replacement.

## 9. Next Recommended Experiments

- Run multi-seed only for rows that beat R++ without diffusion.
- If products remains below target, avoid diffusion and focus on sparse train-label affinity plus target coreset allocation.
- If PrototypeOracleDemand is much better than V2/V3, improve shadow reconstruction before adding model capacity.

## Files

- Small CSV: `experiments\tables\lad_stage_small_seed42.csv`
- Medium CSV: `experiments\tables\lad_stage_medium_seed42.csv`
- Diagnostics CSV: `experiments\tables\lad_stage_diagnostics_seed42.csv`
- Report: `experiments\reports\lad_stage_summary.md`
