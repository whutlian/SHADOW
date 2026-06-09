# Shadow-HGC-L / LAD Stage Summary

## 1. Scope

- Seed policy: single seed `42` only.
- No-diffusion decision: diffusion is not promoted because it caused OOM/resource failures on products and is too expensive for large-scale goals. It remains an appendix diagnostic only.
- Variants: V0 current_best, V1 compiled demand head, V2 compiled demand head + LAD, V3 V2 + boundary-aware prototypes.
- Main matrix ratios: ACM 9.6%; DBLP 0.5% and 6.5%; IMDB 0.5%, 2.5%, 5.0%; ogbn-arxiv/products 6.0% and 12.0%.
- Full-graph ratio sweep: small datasets at 1.2%, 2.4%, 4.8%, 9.6%; ogbn-arxiv/products at 0.05%, 0.25%, 0.5%.
- Result coverage: all LAD-stage tables are summarized here, including 24 small rows, 16 medium rows, 7 diagnostic rows, and 18 full-node sweep rows.

## 2. Code Changes

- Added train-label-only LAD feature computation in `shadow_hgc/features/label_affinity.py`.
- Added compiled demand schema/table helpers in `shadow_hgc/features/compiled_table.py`.
- Added block-gated compiled demand MLP in `shadow_hgc/models/compiled_demand.py`.
- Added boundary-aware prototype helper in `shadow_hgc/prototype/boundary.py`.
- Integrated opt-in LAD/compiled/boundary arguments into `shadow_hgc/pipeline/core.py` without changing the default R-1 path.
- Added LAD scripts under `scripts/run_lad_*.py`, including `scripts/run_lad_full_node_ratio.py`, and tests under `tests/test_*lad*`, `tests/test_compiled_*`, and `tests/test_boundary_*`.
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

## 5. Full-Graph Ratio Sweep

This supplemental LAD-stage sweep reruns no-diffusion V2 (`compiled_plus_lad`) with the budget set by full-graph condensed node ratio, not target prototype ratio. It includes small datasets at `1.2/2.4/4.8/9.6%` and medium datasets at `0.05/0.25/0.5%`, all with seed `42`.

| Dataset | Requested full node ratio | Actual full node ratio | Acc | Macro-F1 | Weighted-F1 | Pred classes | Byte comp | Condensed nodes | Target prototypes | Shadow nodes | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| acm | 1.200% | 1.197% | 0.8116 | 0.8086 | 0.8084 | 3 | 1.1661% | 131 | 52 | 79 | completed |
| acm | 2.400% | 2.404% | 0.8055 | 0.8086 | 0.8078 | 3 | 2.3408% | 263 | 105 | 158 | completed |
| acm | 4.800% | 4.789% | 0.7984 | 0.7987 | 0.7982 | 3 | 4.6612% | 524 | 209 | 315 | completed |
| acm | 9.600% | 9.514% | 0.8546 | 0.8567 | 0.8551 | 3 | 9.2535% | 1041 | 411 | 630 | completed |
| dblp | 1.200% | 1.202% | 0.4732 | 0.4759 | 0.4790 | 4 | 1.1803% | 314 | 157 | 157 | completed |
| dblp | 2.400% | 2.400% | 0.5222 | 0.5020 | 0.5086 | 4 | 2.3538% | 627 | 314 | 313 | completed |
| dblp | 4.800% | 4.715% | 0.4958 | 0.4713 | 0.4793 | 4 | 4.6968% | 1232 | 605 | 627 | completed |
| dblp | 9.600% | 9.117% | 0.4570 | 0.4747 | 0.4823 | 4 | 9.3310% | 2382 | 1128 | 1254 | completed |
| imdb | 1.200% | 1.200% | 0.2552 | 0.1314 | 0.1581 | 5 | 1.2170% | 257 | 128 | 129 | completed |
| imdb | 2.400% | 2.395% | 0.2761 | 0.1005 | 0.1330 | 3 | 2.4293% | 513 | 256 | 257 | completed |
| imdb | 4.800% | 4.748% | 0.0765 | 0.0388 | 0.0234 | 3 | 4.8145% | 1017 | 503 | 514 | completed |
| imdb | 9.600% | 8.730% | 0.1246 | 0.0840 | 0.0821 | 5 | 8.8221% | 1870 | 842 | 1028 | completed |
| ogbn-arxiv | 0.050% | 0.050% | 0.5949 | 0.3285 | 0.5829 | 32 | 0.0390% | 85 | 57 | 28 | completed |
| ogbn-arxiv | 0.250% | 0.250% | 0.6012 | 0.3961 | 0.5911 | 39 | 0.1915% | 423 | 282 | 141 | completed |
| ogbn-arxiv | 0.500% | 0.500% | 0.5977 | 0.4110 | 0.5943 | 38 | 0.3836% | 847 | 565 | 282 | completed |
| ogbn-products | 0.050% | 0.050% | 0.5894 | 0.3270 | 0.5940 | 36 | 0.0130% | 1225 | 817 | 408 | completed |
| ogbn-products | 0.250% | 0.250% | 0.5733 | 0.3033 | 0.5909 | 35 | 0.0647% | 6111 | 4070 | 2041 | completed |
| ogbn-products | 0.500% | 0.497% | 0.5884 | 0.3366 | 0.6142 | 34 | 0.1289% | 12175 | 8093 | 4082 | completed |

Full-graph sweep observations:

- ACM improves at the largest full-node ratio: `9.514%` actual full-node ratio gives `0.8546` accuracy / `0.8567` macro-F1.
- DBLP and IMDB are not helped by LAD V2 under full-node ratio budgeting; DBLP peaks at `0.5222`, and IMDB peaks at `0.2761`.
- ogbn-arxiv remains strong at very small full-node ratios: `0.250%` actual reaches `0.6012` accuracy, and `0.500%` actual reaches `0.4110` macro-F1.
- ogbn-products keeps good macro-F1 at tiny full-node ratios: `0.497%` actual reaches `0.3366` macro-F1, but accuracy remains below the earlier 12% target-ratio V2 row.

## 6. LAD Analysis

- LAD uses training labels only; validation/test labels are not used in label-affinity construction.
- LAD blocks are target-side compiled features, not exposed graph edge types.
- LAD block statistics and learned block gates are logged in per-run JSON files.
- Medium LAD gains are strong: at 12%, ogbn-arxiv V2 improves over V1 from `0.5200` to `0.5968`; ogbn-products V2 improves from `0.4658` to `0.6587`.
- Small LAD is mixed: ACM V2 improves over V1 but remains below the R++ ACM best; DBLP compiled rows are much worse than V0; IMDB improves at 0.5% but does not beat the 2.5% R++ best.

## 7. Boundary Prototype Analysis

- V3 enables boundary-aware prototypes with `boundary_fraction=0.30` and train-only entropy scoring.
- Boundary pool sizes, score stats, and base/boundary prototype counts are logged in V3 JSON files.
- Boundary prototypes are not promoted from this stage. V3 generally underperforms V2, except a small macro-F1 improvement on ogbn-arxiv 6%; it hurts products and IMDB.

## 8. Compression and Resource Accounting

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

## 9. Decision

- Promote LAD for medium no-diffusion experiments and as the replacement signal to study instead of diffusion. Do not promote LAD as a universal small-dataset default.
- Do not promote compiled head alone. V1 usually underperforms V0; its value appears only when paired with LAD on medium datasets.
- Do not promote boundary prototypes as a default. Keep V3 as an ablation until the scoring/allocation strategy is improved.
- Return to large-scale stage only after improving the compiled/prototype path for products, because products accuracy is `0.6587`, below the `0.70` target, despite macro-F1 improving beyond the R++ `0.308` reference.

Direct bottleneck answers:

- Is the bottleneck condensation? Partly yes for arxiv/products: FullDemandTable is higher than condensed V2. For IMDB, the stronger oracle prototype row points more specifically to prototype/shadow reconstruction.
- Is the bottleneck shadow factorization? Yes for IMDB and mildly for arxiv; no for products, where PrototypeOracle and V2 are effectively tied.
- Is the bottleneck training head? Compiled head alone is not enough. V1 is weak, but FullDemandTable shows the same head can be strong when trained on all target rows for arxiv/products, so the head is not the only bottleneck.
- Is LAD useful enough to replace diffusion? For arxiv, yes as a scalable no-diffusion promoted path because V2 reaches `0.5968` and passes the `0.58` gate. For products, LAD improves macro-F1 to `0.3381` but does not reach `0.70` accuracy, so it is promising but not sufficient. For small datasets, no universal replacement.

## 10. Next Recommended Experiments

- Run multi-seed only for rows that beat R++ without diffusion.
- If products remains below target, avoid diffusion and focus on sparse train-label affinity plus target coreset allocation.
- If PrototypeOracleDemand is much better than V2/V3, improve shadow reconstruction before adding model capacity.

## Files

- Small CSV: `experiments\tables\lad_stage_small_seed42.csv`
- Medium CSV: `experiments\tables\lad_stage_medium_seed42.csv`
- Diagnostics CSV: `experiments\tables\lad_stage_diagnostics_seed42.csv`
- Full-node ratio CSV: `experiments\tables\lad_full_node_ratio_seed42.csv`
- Full-node ratio report: `experiments\reports\lad_full_node_ratio_seed42.md`
- Report: `experiments\reports\lad_stage_summary.md`
