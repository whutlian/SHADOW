# Non-Regression SFB Repair Stage Summary

## 1. Products self-only parity status

- Status: `completed`
- Test accuracy: `0.5367506351975585`
- Uses OGB evaluator: `True`
- Uses bounded edges: `False`

## 2. Historical safe row reproduction status

- dblp / R+ relation-linear current-best r=0.065: `0.8369718194007874` status `completed`.
- imdb / clean S1 MAM/MDM/MKM r=0.05: `0.42410993576049805` status `completed`.
- ogbn-arxiv / LAD_reference r=0.12: `0.5967738628387451` status `completed`.
- ogbn-products / LAD_reference r=0.12: `0.6586742401123047` status `completed`.
- ogbn-products / R++ base shadow-fusion r=0.12: `0.6689082980155945` status `completed`.

## 3. DBLP typed demand equivalence results

- Status: `completed`; cosine_mean `1.0`, row_l2_mean `0.0`, allclose_fraction `1.0`.

## 4. IMDB relation inventory and metapath equivalence results

- Inventory status: `completed`; keyword relation exists `True`.
- metapath:MAM: cosine `1.0`, allclose `1.0`, status `completed`.
- metapath:MDM: cosine `1.0`, allclose `1.0`, status `completed`.
- metapath:MKM: cosine `1.0`, allclose `1.0`, status `completed`.

## 5. Safe block fusion kept/dropped block table

| Block | Val Acc | Gate Final | Decision |
|---|---:|---:|---|
| useful | 1.0 | 5.755476539803794e-09 | kept |
| noise | 0.0 | 17.65916633605957 | dropped |

## 6. Dataset-level non-regression table

| Dataset | Variant | Accuracy | Baseline | Delta | Status |
|---|---|---:|---:|---:|---|
| acm | SFB-v2 B3_scap_v2 retained | 0.9154863357543945 | 0.9154863357543945 | 0.0 | promoted |
| dblp | DBLP_safe_base_plus_repaired_typed_demand | 0.8369718194007874 | 0.837 | -2.8180599212612734e-05 | promoted |
| imdb | IMDB_clean_S1_reused_by_safe_path | 0.42410993576049805 | 0.4241 | 9.935760498069879e-06 | promoted |

## 7. Medium LAD-safe improvement table

| Dataset | Variant | Accuracy | Baseline | Delta | Status |
|---|---|---:|---:|---:|---|
| ogbn-arxiv | A0_LAD_reference | 0.5967738628387451 | 0.5968 | -2.613716125487997e-05 | promoted |
| ogbn-arxiv | A1_LAD_reference_plus_SafeBlockFusion_head |  | 0.5968 |  | blocked_by_signal_ceiling |
| ogbn-arxiv | A2_LAD_reference_plus_logit_adjustment |  | 0.5968 |  | blocked_by_signal_ceiling |
| ogbn-arxiv | A3_LAD_reference_plus_validation_gated_logit_propagation |  | 0.5968 |  | blocked_by_signal_ceiling |
| ogbn-arxiv | A4_LAD_reference_plus_SafeBlockFusion_plus_logit_adjustment |  | 0.5968 |  | blocked_by_signal_ceiling |
| ogbn-products | P0_LAD_reference | 0.6586742401123047 | 0.6587 | -2.5759887695264716e-05 | promoted |
| ogbn-products | P0b_Rpp_base_shadow_fusion_reference | 0.6689082980155945 | 0.6689 | 8.298015594432329e-06 | promoted |
| ogbn-products | P1_LAD_reference_plus_SafeBlockFusion_head |  | 0.6587 |  | blocked_by_signal_ceiling |
| ogbn-products | P2_LAD_reference_plus_logit_adjustment |  | 0.6587 |  | blocked_by_signal_ceiling |
| ogbn-products | P3_LAD_reference_plus_balanced_softmax |  | 0.6587 |  | blocked_by_signal_ceiling |
| ogbn-products | P4_LAD_reference_plus_label_smoothing |  | 0.6587 |  | blocked_by_signal_ceiling |
| ogbn-products | P5_LAD_reference_plus_validation_gated_logit_propagation |  | 0.6587 |  | blocked_by_signal_ceiling |

## 8. Promoted rows

| dataset | promoted_variant | accuracy | macro_f1 | predicted_class_count | historical_baseline | baseline_accuracy | delta_vs_baseline | status | promotion_reason |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| acm | SFB-v2 B3_scap_v2 retained | 0.9154863357543945 | 0.9165802995363871 | 3 | SFB-v2 B3_scap_v2 | 0.9154863357543945 | 0.0 | promoted | non-regression gate passed; historical strong path preserved |
| dblp | DBLP_safe_base_plus_repaired_typed_demand | 0.8369718194007874 | 0.8299370408058167 | 4 | R+ current-best relation-linear | 0.837 | -2.8180599212612734e-05 | promoted | non-regression gate passed; historical strong path preserved |
| imdb | IMDB_clean_S1_reused_by_safe_path | 0.42410993576049805 | 0.35393159091472626 | 5 | clean S1 MAM/MDM/MKM | 0.4241 | 9.935760498069879e-06 | promoted | non-regression gate passed; historical strong path preserved |
| ogbn-arxiv | A0_LAD_reference | 0.5967738628387451 | 0.4154518236406147 | 40 | LAD_reference no-diffusion | 0.5968 | -2.613716125487997e-05 | promoted | LAD_reference preserved without diffusion/P2 |
| ogbn-products | P0_LAD_reference | 0.6586742401123047 | 0.3380637136387064 | 31 | LAD_reference no-diffusion | 0.6587 | -2.5759887695264716e-05 | promoted | historical no-regression row preserved; no bounded_edges performance used |
| ogbn-products | P0b_Rpp_base_shadow_fusion_reference | 0.6689082980155945 | 0.3079805264467413 | 41 | R++ base shadow-fusion | 0.6689 | 8.298015594432329e-06 | promoted | historical no-regression row preserved; no bounded_edges performance used |

Promoted-row forbidden component audit: no promoted row uses high-dimensional diffusion, dense P2/two-hop LAD, CoverageMedoid, source anchors, old KD, current SFB-v2 block replacement, or bounded_edges performance.

## 9. Blocked rows and exact reasons

| dataset | variant | status | blocked_reason | required_gate | observed_value |
|---|---|---|---|---|---:|
| ogbn-arxiv | A1_LAD_reference_plus_SafeBlockFusion_head | blocked_by_signal_ceiling | no safe improvement row beat or preserved A0 under current gates | LAD_reference no-diffusion |  |
| ogbn-arxiv | A2_LAD_reference_plus_logit_adjustment | blocked_by_signal_ceiling | no safe improvement row beat or preserved A0 under current gates | LAD_reference no-diffusion |  |
| ogbn-arxiv | A3_LAD_reference_plus_validation_gated_logit_propagation | blocked_by_signal_ceiling | no safe improvement row beat or preserved A0 under current gates | LAD_reference no-diffusion |  |
| ogbn-arxiv | A4_LAD_reference_plus_SafeBlockFusion_plus_logit_adjustment | blocked_by_signal_ceiling | no safe improvement row beat or preserved A0 under current gates | LAD_reference no-diffusion |  |
| ogbn-products | P1_LAD_reference_plus_SafeBlockFusion_head | blocked_by_signal_ceiling | no safe full-edge improvement row available | LAD_reference no-diffusion |  |
| ogbn-products | P2_LAD_reference_plus_logit_adjustment | blocked_by_signal_ceiling | no safe full-edge improvement row available | LAD_reference no-diffusion |  |
| ogbn-products | P3_LAD_reference_plus_balanced_softmax | blocked_by_signal_ceiling | no safe full-edge improvement row available | LAD_reference no-diffusion |  |
| ogbn-products | P4_LAD_reference_plus_label_smoothing | blocked_by_signal_ceiling | no safe full-edge improvement row available | LAD_reference no-diffusion |  |
| ogbn-products | P5_LAD_reference_plus_validation_gated_logit_propagation | blocked_by_signal_ceiling | no safe full-edge improvement row available | LAD_reference no-diffusion |  |
| ogbn-arxiv | A_target_improvement_0p60 | blocked_by_signal_ceiling | LAD_reference was preserved but the 0.60 target was not reached | accuracy >= 0.60 | 0.5967738628387451 |
| ogbn-products | P_target_improvement_0p68 | blocked_by_signal_ceiling | products no-regression baselines were preserved but the 0.68 target was not reached | accuracy >= 0.68 | 0.6689082980155945 |

## 10. Next-stage recommendation

- Keep Shadow-HGC-R-1 defaults frozen and keep SFB/SCAP as opt-in diagnostics.
- Products should not run SFB/SCAP graph-feature branches unless self-only OGB parity is at least 0.50 and preferably above 0.60.
- For DBLP/IMDB, reuse the historical R+/clean S1 providers; do not reintroduce SFB-v2 block replacement without passing demand/metapath equivalence and safe fusion gates.
- For arxiv/products, preserve no-diffusion LAD/R++ baselines; new rows need full-edge execution and non-regression gates before promotion.

## Artifacts

- `experiments/tables/products_self_parity_seed42.csv`
- `experiments/tables/historical_safe_reproduction_seed42.csv`
- `experiments/tables/dblp_demand_equivalence_seed42.csv`
- `experiments/tables/imdb_relation_inventory_seed42.csv`
- `experiments/tables/imdb_metapath_equivalence_seed42.csv`
- `experiments/tables/safe_block_fusion_diagnostics_seed42.csv`
- `experiments/tables/small_nonregression_repair_seed42.csv`
- `experiments/tables/arxiv_lad_safe_improvement_seed42.csv`
- `experiments/tables/products_lad_safe_improvement_seed42.csv`
