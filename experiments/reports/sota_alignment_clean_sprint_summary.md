# SOTA Alignment Clean Sprint Summary

## Scope

- Seed policy: single seed `42` only.
- Default method remains frozen as `Shadow-HGC-R-1`; all SOTA alignment paths are explicit scripts/diagnostics.
- Diffusion, CoverageMedoid, source anchors, and old KD are not promoted.
- Invalid historical rows are retained in audit artifacts but excluded from best-row summaries.

## Code Changes

- Added hard audit gates in `shadow_hgc/audit/*` and read-only `scripts/run_sota_audit.py`.
- Added schema-default meta-path specs and DBLP schema audit.
- Added Path-LAD v2 diagnostics, `P2` target-target Path-LAD support, two-hop LAD utility, teacher-demand herding selector, and KD v2 gate/loss.
- Added actual SeHGNNLite target-row/prototype training utilities for clean small/fullgraph audits.
- Added clean experiment scripts for fullgraph backbone, clean small, medium no-diffusion refine, gated teacher/KD diagnostics, and this summary.
- Updated small dataset YAML configs to match the loader-exposed relations.

## Pytest

148 passed in 75.85s (0:01:15)

## Hard Audit Status

| status | count |
| --- | --- |
| completed | 63 |
| invalid_config | 43 |
| oom | 5 |
| timeout_dropped | 9 |

## Fullgraph Backbone Audit

| dataset | variant | accuracy | macro_f1 | target_gate | gate_passed | blocked_by_fullgraph_backbone | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acm | fullgraph_sehgnn_lite | 0.9117091298103333 | 0.912367602189382 | 0.9 | True | False | completed |
| dblp | fullgraph_sehgnn_lite | 0.7880281805992126 | 0.7809299379587173 | 0.88 | False | True | completed |
| imdb | fullgraph_sehgnn_lite | 0.40818238258361816 | 0.36262465715408326 | 0.55 | False | True | completed |
| ogbn-arxiv | fullgraph_no_diffusion_lad_table_teacher | 0.6615641117095947 | 0.4024657105095685 | 0.65 | True | False | completed_existing_diagnostic |
| ogbn-products | fullgraph_no_diffusion_lad_table_teacher | 0.6884398460388184 | 0.33906127453008866 | 0.7 | False | True | completed_existing_diagnostic |

Interpretation: ACM and arxiv passed the first backbone gates. DBLP, IMDB, and products are marked as backbone/data constrained for SOTA chasing in this sprint.

## DBLP Schema Audit

| target_type | label_node_type | apa_available | computed_metapath_blocks | skipped_metapath_blocks | hard_requirements_passed | notes |
| --- | --- | --- | --- | --- | --- | --- |
| author | author | True | ["APA"] | ["APVPA", "APTPA", "APCPA"] | True | APVPA/APTPA require non-target paper-venue/paper-term edges; current small loader keeps incoming-to-target relations only. |

## Clean Small Results

Best completed clean row per small dataset:
| dataset | variant | requested_ratio | accuracy | macro_f1 | predicted_class_count | total_condensed_node_ratio | model_type | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | S1_clean_metapath_sehgnn | 0.12 | 0.8937677145004272 | 0.8940866192181905 | 3 | 0.0097788338512155 | sehgnn_lite | completed |
| dblp | S0_current_best | 0.096 | 0.7845070362091064 | 0.7770234197378159 | 4 | 0.008955909369259033 | relation_linear | completed |
| imdb | S1_clean_MAM_MDM_MKM | 0.05 | 0.42410993576049805 | 0.35393159091472626 | 5 | 0.0032212885154061623 | sehgnn_lite | completed |

All clean small rows:
| dataset | variant | requested_ratio | accuracy | macro_f1 | predicted_class_count | total_condensed_node_ratio | model_type | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | S1_clean_metapath_sehgnn | 0.048 | 0.8432483673095703 | 0.8443612655003866 | 3 | 0.004021202705172729 | sehgnn_lite | completed |
| acm | S1_clean_metapath_sehgnn | 0.096 | 0.8800755143165588 | 0.8795819679896036 | 3 | 0.007768232498629135 | sehgnn_lite | completed |
| acm | S1_clean_metapath_sehgnn | 0.12 | 0.8937677145004272 | 0.8940866192181905 | 3 | 0.0097788338512155 | sehgnn_lite | completed |
| acm | S1_clean_metapath_sehgnn | 0.15 | 0.8706326484680176 | 0.869088351726532 | 3 | 0.011698044233229756 | sehgnn_lite | completed |
| dblp | S0_current_best | 0.005 | 0.7838028073310852 | 0.7758937478065491 | 4 | 0.001224739742804654 | relation_linear | completed |
| dblp | S1_clean_APA_sehgnn | 0.005 | 0.7626760601997375 | 0.7586506754159927 | 4 | 0.000612369871402327 | sehgnn_lite | completed |
| dblp | S0_current_best | 0.065 | 0.7785211205482483 | 0.7709946632385254 | 4 | 0.006047152480097979 | relation_linear | completed |
| dblp | S1_clean_APA_sehgnn | 0.065 | 0.7700704336166382 | 0.7602305710315704 | 4 | 0.0030235762400489894 | sehgnn_lite | completed |
| dblp | S0_current_best | 0.096 | 0.7845070362091064 | 0.7770234197378159 | 4 | 0.008955909369259033 | relation_linear | completed |
| dblp | S1_clean_APA_sehgnn | 0.096 | 0.7644366025924683 | 0.7589197754859924 | 4 | 0.004477954684629516 | sehgnn_lite | completed |
| imdb | Rpp_shadow_fusion_class_balanced_reference | 0.005 | 0.3425983786582947 | 0.3311773508787155 | 5 | 0.005929038281979458 | shadow_fusion | completed |
| imdb | S1_clean_MAM_MDM_MKM | 0.005 | 0.37164270877838135 | 0.34688123464584353 | 5 | 0.0009337068160597573 | sehgnn_lite | completed |
| imdb | PathLAD_v2_only | 0.005 | 0.35352903604507446 | 0.34126800000667573 | 5 | 0.0009337068160597573 | sehgnn_lite | completed |
| imdb | PathLAD_v2_plus_shadow_fusion | 0.005 | 0.3279200494289398 | 0.31737546622753143 | 5 | 0.0009337068160597573 | sehgnn_lite | completed |
| imdb | Rpp_shadow_fusion_class_balanced_reference | 0.025 | 0.34134915471076965 | 0.3266520440578461 | 5 | 0.011064425770308124 | shadow_fusion | completed |
| imdb | S1_clean_MAM_MDM_MKM | 0.025 | 0.3863210380077362 | 0.3676772892475128 | 5 | 0.0015873015873015873 | sehgnn_lite | completed |
| imdb | PathLAD_v2_only | 0.025 | 0.35227981209754944 | 0.3251879423856735 | 5 | 0.0015873015873015873 | sehgnn_lite | completed |
| imdb | PathLAD_v2_plus_shadow_fusion | 0.025 | 0.3113678991794586 | 0.3068678677082062 | 5 | 0.0015873015873015873 | sehgnn_lite | completed |
| imdb | Rpp_shadow_fusion_class_balanced_reference | 0.05 | 0.3504059910774231 | 0.33577802777290344 | 5 | 0.022549019607843137 | shadow_fusion | completed |
| imdb | S1_clean_MAM_MDM_MKM | 0.05 | 0.42410993576049805 | 0.35393159091472626 | 5 | 0.0032212885154061623 | sehgnn_lite | completed |
| imdb | PathLAD_v2_only | 0.05 | 0.367582768201828 | 0.3188542157411575 | 5 | 0.0032212885154061623 | sehgnn_lite | completed |
| imdb | PathLAD_v2_plus_shadow_fusion | 0.05 | 0.3372891843318939 | 0.30090869665145875 | 5 | 0.0032212885154061623 | sehgnn_lite | completed |

## Medium No-Diffusion Refine

Best completed medium row per dataset:
| dataset | variant | requested_ratio | accuracy | macro_f1 | predicted_class_count | total_condensed_node_ratio | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | LAD_reference | 0.12 | 0.5967738628387451 | 0.4154518236406147 | 40 | 0.06941532865249818 | completed |
| ogbn-products | LAD_reference | 0.12 | 0.6586742401123047 | 0.3380637136387064 | 31 | 0.009596048066396927 | completed |

All medium rows:
| dataset | variant | requested_ratio | accuracy | macro_f1 | predicted_class_count | two_hop_lad_blocks | status | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | LAD_reference | 0.06 | 0.590210497379303 | 0.40769136818125845 | 39 | [] | completed |  |
| ogbn-arxiv | LAD_plus_two_hop_LAD | 0.06 | 0.33508220314979553 | 0.1428937314078212 | 34 | ["P2"] | completed |  |
| ogbn-arxiv | LAD_plus_two_hop_LAD_plus_lad_fusion_head | 0.06 | 0.3188897669315338 | 0.09305294038495049 | 24 | ["P2"] | completed |  |
| ogbn-arxiv | LAD_reference | 0.12 | 0.5967738628387451 | 0.4154518236406147 | 40 | [] | completed |  |
| ogbn-arxiv | LAD_plus_two_hop_LAD | 0.12 | 0.22848384082317352 | 0.05985004580579698 | 21 | ["P2"] | completed |  |
| ogbn-arxiv | LAD_plus_two_hop_LAD_plus_lad_fusion_head | 0.12 | 0.22665268182754517 | 0.019331736117601396 | 2 | ["P2"] | completed |  |
| ogbn-products | LAD_reference | 0.06 | 0.6223331689834595 | 0.3307438173598828 | 32 | [] | completed |  |
| ogbn-products | LAD_plus_two_hop_LAD | 0.06 |  |  |  | ["P2"] | oom | [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 23259036640 bytes. |
| ogbn-products | LAD_plus_two_hop_LAD_plus_lad_fusion_head | 0.06 |  |  |  | ["P2"] | oom | [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 23259036640 bytes. |
| ogbn-products | LAD_plus_balanced_softmax | 0.06 |  |  |  | ["P2"] | oom | [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 23259036640 bytes. |
| ogbn-products | LAD_reference | 0.12 | 0.6586742401123047 | 0.3380637136387064 | 31 | [] | completed |  |
| ogbn-products | LAD_plus_two_hop_LAD | 0.12 |  |  |  | ["P2"] | oom | [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 23259036640 bytes. |
| ogbn-products | LAD_plus_two_hop_LAD_plus_lad_fusion_head | 0.12 |  |  |  | ["P2"] | oom | [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 23259036640 bytes. |
| ogbn-products | LAD_plus_balanced_softmax | 0.12 |  |  |  | ["P2"] | timeout_dropped | products medium refine materialized after 30-minute timeout; row did not finish before watchdog |

## Teacher Herding / KD Gates

| dataset | variant | status | kd_gate_passed | kd_skip_reason | teacher_type |
| --- | --- | --- | --- | --- | --- |
| acm | teacher_demand_herding | skipped_blocked_by_teacher_or_backbone | False | fullgraph_or_teacher_gate_not_passed_in_clean_sprint | none |
| acm | kd_v2 | skipped_blocked_by_teacher_or_backbone | False | fullgraph_or_teacher_gate_not_passed_in_clean_sprint | none |
| dblp | teacher_demand_herding | skipped_blocked_by_teacher_or_backbone | False | fullgraph_or_teacher_gate_not_passed_in_clean_sprint | none |
| dblp | kd_v2 | skipped_blocked_by_teacher_or_backbone | False | fullgraph_or_teacher_gate_not_passed_in_clean_sprint | none |
| imdb | teacher_demand_herding | skipped_blocked_by_teacher_or_backbone | False | fullgraph_or_teacher_gate_not_passed_in_clean_sprint | none |
| imdb | kd_v2 | skipped_blocked_by_teacher_or_backbone | False | fullgraph_or_teacher_gate_not_passed_in_clean_sprint | none |
| ogbn-arxiv | teacher_demand_herding | skipped_blocked_by_teacher_or_backbone | False | fullgraph_or_teacher_gate_not_passed_in_clean_sprint | none |
| ogbn-arxiv | kd_v2 | skipped_blocked_by_teacher_or_backbone | False | fullgraph_or_teacher_gate_not_passed_in_clean_sprint | none |
| ogbn-products | teacher_demand_herding | skipped_blocked_by_teacher_or_backbone | False | fullgraph_or_teacher_gate_not_passed_in_clean_sprint | none |
| ogbn-products | kd_v2 | skipped_blocked_by_teacher_or_backbone | False | fullgraph_or_teacher_gate_not_passed_in_clean_sprint | none |

## Invalid Examples

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

## Conclusions

- Promote: audit gates, schema-default clean SeHGNNLite for ACM, DBLP APA schema audit, and no-diffusion LAD reference rows.
- Keep as diagnostic: Path-LAD v2 feature blocks on IMDB; they are train-label-only and valid but did not beat clean MAM/MDM/MKM.
- Drop from promoted path: CoverageMedoid, old KD, source anchors, products P2 LAD as currently implemented, and high-dimensional diffusion.
- Bottlenecks: DBLP/IMDB fullgraph backbone capacity is below requested gates; arxiv two-hop LAD caused large regression and class collapse; products two-hop LAD hit CPU OOM due a 23GB allocation path and one row timed out.

## Artifact Files

- `experiments/tables/sota_audit_seed42.csv`
- `experiments/tables/fullgraph_backbone_audit_seed42.csv`
- `experiments/tables/dblp_schema_audit_seed42.csv`
- `experiments/tables/sota_clean_small_seed42.csv`
- `experiments/tables/medium_no_diffusion_refine_seed42.csv`
- `experiments/tables/teacher_herding_kd_gated_seed42.csv`
