# Products LAD Safe Improvement Seed 42

| dataset | variant | accuracy | macro_f1 | predicted_class_count | historical_baseline | baseline_accuracy | delta_vs_baseline | status | promotion_reason | blocked_reason | source_log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ogbn-products | P0_LAD_reference | 0.6586742401123047 | 0.3380637136387064 | 31 | LAD_reference no-diffusion | 0.6587 | -2.5759887695264716e-05 | promoted | historical no-regression row preserved; no bounded_edges performance used |  | experiments\logs\lad_stage_medium_seed42\ogbn-products_V2_compiled_plus_lad_r0p12_seed42.json |
| ogbn-products | P0b_Rpp_base_shadow_fusion_reference | 0.6689082980155945 | 0.3079805264467413 | 41 | R++ base shadow-fusion | 0.6689 | 8.298015594432329e-06 | promoted | historical no-regression row preserved; no bounded_edges performance used |  | experiments\logs\products_streaming_diffusion_seed42\ogbn-products_base_sqrt_weighted_r0p12_seed42.json |
| ogbn-products | P1_LAD_reference_plus_SafeBlockFusion_head |  |  |  | LAD_reference no-diffusion | 0.6587 |  | blocked_by_signal_ceiling |  | no safe full-edge improvement row available |  |
| ogbn-products | P2_LAD_reference_plus_logit_adjustment |  |  |  | LAD_reference no-diffusion | 0.6587 |  | blocked_by_signal_ceiling |  | no safe full-edge improvement row available |  |
| ogbn-products | P3_LAD_reference_plus_balanced_softmax |  |  |  | LAD_reference no-diffusion | 0.6587 |  | blocked_by_signal_ceiling |  | no safe full-edge improvement row available |  |
| ogbn-products | P4_LAD_reference_plus_label_smoothing |  |  |  | LAD_reference no-diffusion | 0.6587 |  | blocked_by_signal_ceiling |  | no safe full-edge improvement row available |  |
| ogbn-products | P5_LAD_reference_plus_validation_gated_logit_propagation |  |  |  | LAD_reference no-diffusion | 0.6587 |  | blocked_by_signal_ceiling |  | no safe full-edge improvement row available |  |

- CSV: `experiments\tables\products_lad_safe_improvement_seed42.csv`
