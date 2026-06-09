# SOTA Alignment Audit Seed 42

This is a read-only audit over historical and clean SOTA JSON/CSV artifacts. Invalid rows keep their historical logs but are excluded from best-row summaries.

## Valid Best Rows
| dataset | variant | accuracy | macro_f1 | model_type | total_condensed_node_ratio |
| --- | --- | --- | --- | --- | --- |
| acm | S1_clean_metapath_sehgnn | 0.8937677145004272 | 0.8940866192181905 | sehgnn_lite | 0.0097788338512155 |
| dblp | S0_current_best | 0.8056337833404541 | 0.7980579286813736 | relation_linear | 0.013433864053888548 |
| imdb | S1_clean_MAM_MDM_MKM | 0.42410993576049805 | 0.35393159091472626 | sehgnn_lite | 0.0032212885154061623 |
| ogbn-arxiv | LAD_reference | 0.5967738628387451 | 0.4154518236406147 | compiled_demand_mlp | 0.06941532865249818 |
| ogbn-products | LAD_reference | 0.6586742401123047 | 0.3380637136387064 | compiled_demand_mlp | 0.009596048066396927 |

## Invalid / Failed Rows
| dataset | variant | status | invalid_reasons | source_log |
| --- | --- | --- | --- | --- |
| acm | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\acm_S1_sehgnn_lite_metapath_r0p012_seed42.json |
| acm | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\acm_S3_path_lad_source_anchor_r0p012_seed42.json |
| acm | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\acm_S4_teacher_kd_r0p012_seed42.json |
| acm | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\acm_S1_sehgnn_lite_metapath_r0p024_seed42.json |
| acm | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\acm_S3_path_lad_source_anchor_r0p024_seed42.json |
| acm | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\acm_S4_teacher_kd_r0p024_seed42.json |
| acm | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\acm_S1_sehgnn_lite_metapath_r0p048_seed42.json |
| acm | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\acm_S3_path_lad_source_anchor_r0p048_seed42.json |
| acm | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\acm_S4_teacher_kd_r0p048_seed42.json |
| acm | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\acm_S1_sehgnn_lite_metapath_r0p096_seed42.json |
| acm | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\acm_S3_path_lad_source_anchor_r0p096_seed42.json |
| acm | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\acm_S4_teacher_kd_r0p096_seed42.json |
| dblp | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\dblp_S1_sehgnn_lite_metapath_r0p012_seed42.json |
| dblp | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\dblp_S3_path_lad_source_anchor_r0p012_seed42.json |
| dblp | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\dblp_S4_teacher_kd_r0p012_seed42.json |
| dblp | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\dblp_S1_sehgnn_lite_metapath_r0p024_seed42.json |
| dblp | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\dblp_S3_path_lad_source_anchor_r0p024_seed42.json |
| dblp | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\dblp_S4_teacher_kd_r0p024_seed42.json |
| dblp | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\dblp_S1_sehgnn_lite_metapath_r0p048_seed42.json |
| dblp | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\dblp_S3_path_lad_source_anchor_r0p048_seed42.json |
| dblp | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\dblp_S4_teacher_kd_r0p048_seed42.json |
| dblp | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\dblp_S1_sehgnn_lite_metapath_r0p096_seed42.json |
| dblp | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\dblp_S3_path_lad_source_anchor_r0p096_seed42.json |
| dblp | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\dblp_S4_teacher_kd_r0p096_seed42.json |
| imdb | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\imdb_S1_sehgnn_lite_metapath_r0p012_seed42.json |
| imdb | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\imdb_S3_path_lad_source_anchor_r0p012_seed42.json |
| imdb | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\imdb_S4_teacher_kd_r0p012_seed42.json |
| imdb | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\imdb_S1_sehgnn_lite_metapath_r0p024_seed42.json |
| imdb | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\imdb_S3_path_lad_source_anchor_r0p024_seed42.json |
| imdb | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\imdb_S4_teacher_kd_r0p024_seed42.json |
| imdb | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\imdb_S1_sehgnn_lite_metapath_r0p048_seed42.json |
| imdb | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\imdb_S3_path_lad_source_anchor_r0p048_seed42.json |
| imdb | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\imdb_S4_teacher_kd_r0p048_seed42.json |
| imdb | S1_sehgnn_lite_metapath | invalid_config | ['sehgnn_or_metapath_requires_model_type_sehgnn_lite'] | experiments\logs\sota_small_seed42\imdb_S1_sehgnn_lite_metapath_r0p096_seed42.json |
| imdb | S3_path_lad_source_anchor | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_small_seed42\imdb_S3_path_lad_source_anchor_r0p096_seed42.json |
| imdb | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_small_seed42\imdb_S4_teacher_kd_r0p096_seed42.json |
| ogbn-arxiv | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_medium_seed42\ogbn-arxiv_S4_teacher_kd_fullnode_r0p0005_seed42.json |
| ogbn-arxiv | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_medium_seed42\ogbn-arxiv_S4_teacher_kd_fullnode_r0p0025_seed42.json |
| ogbn-arxiv | S4_teacher_kd | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_medium_seed42\ogbn-arxiv_S4_teacher_kd_fullnode_r0p005_seed42.json |
| ogbn-products | S2_coverage_medoids | timeout_dropped | ['status_not_completed:timeout_dropped'] | experiments\logs\sota_medium_seed42\ogbn-products_S2_coverage_medoids_fullnode_r0p0005_seed42.json |
| ogbn-products | S4_teacher_kd | timeout_dropped | ['status_not_completed:timeout_dropped'] | experiments\logs\sota_medium_seed42\ogbn-products_S4_teacher_kd_fullnode_r0p0005_seed42.json |
| ogbn-products | S0_current_best | timeout_dropped | ['status_not_completed:timeout_dropped'] | experiments\logs\sota_medium_seed42\ogbn-products_S0_current_best_fullnode_r0p0025_seed42.json |
| ogbn-products | S2_coverage_medoids | timeout_dropped | ['status_not_completed:timeout_dropped'] | experiments\logs\sota_medium_seed42\ogbn-products_S2_coverage_medoids_fullnode_r0p0025_seed42.json |
| ogbn-products | S4_teacher_kd | timeout_dropped | ['status_not_completed:timeout_dropped'] | experiments\logs\sota_medium_seed42\ogbn-products_S4_teacher_kd_fullnode_r0p0025_seed42.json |
| ogbn-products | S0_current_best | timeout_dropped | ['status_not_completed:timeout_dropped'] | experiments\logs\sota_medium_seed42\ogbn-products_S0_current_best_fullnode_r0p005_seed42.json |
| ogbn-products | S2_coverage_medoids | timeout_dropped | ['status_not_completed:timeout_dropped'] | experiments\logs\sota_medium_seed42\ogbn-products_S2_coverage_medoids_fullnode_r0p005_seed42.json |
| ogbn-products | S4_teacher_kd | timeout_dropped | ['status_not_completed:timeout_dropped'] | experiments\logs\sota_medium_seed42\ogbn-products_S4_teacher_kd_fullnode_r0p005_seed42.json |
| imdb | PathLAD-off | invalid_config | ['path_lad_blocks_empty', 'path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_diagnostics_seed42\imdb_PathLAD-off_r0p048_seed42.json |
| imdb | PathLAD-on | invalid_config | ['path_lad_row_normalization_missing', 'path_lad_hub_clipping_missing'] | experiments\logs\sota_diagnostics_seed42\imdb_PathLAD-on_r0p048_seed42.json |
| imdb | KD-off | invalid_config | ['kd_teacher_type_missing', 'teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_diagnostics_seed42\imdb_KD-off_r0p048_seed42.json |
| imdb | KD-on | invalid_config | ['teacher_train_acc_missing', 'teacher_val_acc_missing', 'ce_loss_missing', 'kd_loss_missing'] | experiments\logs\sota_diagnostics_seed42\imdb_KD-on_r0p048_seed42.json |
| ogbn-products | LAD_plus_two_hop_LAD | oom | ['status_not_completed:oom'] | experiments\logs\medium_no_diffusion_refine_seed42\ogbn-products_LAD_plus_two_hop_LAD_r0p06_seed42.json |
| ogbn-products | LAD_plus_two_hop_LAD_plus_lad_fusion_head | oom | ['status_not_completed:oom'] | experiments\logs\medium_no_diffusion_refine_seed42\ogbn-products_LAD_plus_two_hop_LAD_plus_lad_fusion_head_r0p06_seed42.json |
| ogbn-products | LAD_plus_balanced_softmax | oom | ['status_not_completed:oom'] | experiments\logs\medium_no_diffusion_refine_seed42\ogbn-products_LAD_plus_balanced_softmax_r0p06_seed42.json |
| ogbn-products | LAD_plus_two_hop_LAD | oom | ['status_not_completed:oom'] | experiments\logs\medium_no_diffusion_refine_seed42\ogbn-products_LAD_plus_two_hop_LAD_r0p12_seed42.json |
| ogbn-products | LAD_plus_two_hop_LAD_plus_lad_fusion_head | oom | ['status_not_completed:oom'] | experiments\logs\medium_no_diffusion_refine_seed42\ogbn-products_LAD_plus_two_hop_LAD_plus_lad_fusion_head_r0p12_seed42.json |
| ogbn-products | LAD_plus_balanced_softmax | timeout_dropped | ['status_not_completed:timeout_dropped'] | experiments\logs\medium_no_diffusion_refine_seed42\ogbn-products_LAD_plus_balanced_softmax_r0p12_seed42.json |

## Gate Notes

- SeHGNN/meta-path rows require actual `model_type=sehgnn_lite`, non-empty blocks, dims, feature block list, and block norm source.
- KD rows require teacher train/val quality, predicted class count, temperature/lambda, and separate CE/KD losses.
- Path-LAD rows require train-label-only, row normalization, leave-one-out, and hub clipping diagnostics.

- CSV: `experiments\tables\sota_audit_seed42.csv`
