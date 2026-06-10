# T24 Products SFT Recovery

- Train mode: `True`
- Proxy rows are not used for promotion; rows are built from memmap SFT block signatures.

| requested_full_node_ratio | method | status | accuracy | macro_f1 | actual_full_node_ratio | promotion_status | promotion_reason |
|---|---|---|---|---|---|---|---|
| 0.0005 | P0_identity_replay | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | 0.0005001982418338044 | not_promoted | identity replay is reference only |
| 0.0005 | P2_shadow_condensed_centroid_b1 | completed_streaming | 0.08498701589767434 | 0.06907466230306467 | 0.0005001982418338044 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.0005 | P3_shadow_condensed_medoid_b1 | completed_streaming | 0.1280033220504715 | 0.051096553433580785 | 0.0005001982418338044 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.0005 | P4_shadow_condensed_herding_b1 | completed_streaming | 0.18752007938218537 | 0.12994169304363654 | 0.0005001982418338044 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.0005 | P5_shadow_condensed_hybrid_b1 | completed_streaming | 0.14223274144623968 | 0.11243063872158078 | 0.0005001982418338044 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.0005 | P6_best_shadow_b2_ablation | completed_derived_ablation | 0.18752007938218537 | 0.12994169304363654 | 0.0005001982418338044 | not_promoted | derived ablation rows are not promoted |
| 0.0005 | P7_best_shadow_ks_ablation | completed_derived_ablation | 0.18752007938218537 | 0.12994169304363654 | 0.0005001982418338044 | not_promoted | derived ablation rows are not promoted |
| 0.001 | P0_identity_replay | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | 0.0009999881585722342 | not_promoted | identity replay is reference only |
| 0.001 | P2_shadow_condensed_centroid_b1 | completed_streaming | 0.20523015095176836 | 0.09376067105149587 | 0.0009999881585722342 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.001 | P3_shadow_condensed_medoid_b1 | completed_streaming | 0.1352307699954498 | 0.09419741341886802 | 0.0009999881585722342 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.001 | P4_shadow_condensed_herding_b1 | completed_streaming | 0.15243385834563514 | 0.07523902199709431 | 0.0009999881585722342 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.001 | P5_shadow_condensed_hybrid_b1 | completed_streaming | 0.14956953871304884 | 0.09138454469016156 | 0.0009999881585722342 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.001 | P6_best_shadow_b2_ablation | completed_derived_ablation | 0.20523015095176836 | 0.09376067105149587 | 0.0009999881585722342 | not_promoted | derived ablation rows are not promoted |
| 0.001 | P7_best_shadow_ks_ablation | completed_derived_ablation | 0.20523015095176836 | 0.09376067105149587 | 0.0009999881585722342 | not_promoted | derived ablation rows are not promoted |
| 0.0025 | P0_identity_replay | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | 0.0025001745589782725 | not_promoted | identity replay is reference only |
| 0.0025 | P2_shadow_condensed_centroid_b1 | completed_streaming | 0.13638074530148106 | 0.11768598356857071 | 0.0024969079582152763 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.0025 | P3_shadow_condensed_medoid_b1 | completed_streaming | 0.174593814714352 | 0.11342605750338396 | 0.0024969079582152763 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.0025 | P4_shadow_condensed_herding_b1 | completed_streaming | 0.10872711515251746 | 0.08886849416925892 | 0.0024969079582152763 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.0025 | P5_shadow_condensed_hybrid_b1 | completed_streaming | 0.14516664701090015 | 0.1074883741996578 | 0.002496091308024527 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.0025 | P6_best_shadow_b2_ablation | completed_derived_ablation | 0.174593814714352 | 0.11342605750338396 | 0.0024969079582152763 | not_promoted | derived ablation rows are not promoted |
| 0.0025 | P7_best_shadow_ks_ablation | completed_derived_ablation | 0.174593814714352 | 0.11342605750338396 | 0.0024969079582152763 | not_promoted | derived ablation rows are not promoted |
| 0.005 | P0_identity_replay | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | 0.00499994079286117 | not_promoted | identity replay is reference only |
| 0.005 | P2_shadow_condensed_centroid_b1 | completed_streaming | 0.2740235263710349 | 0.179915990105431 | 0.004982382813760066 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.005 | P3_shadow_condensed_medoid_b1 | completed_streaming | 0.27363176661059124 | 0.20221363326640582 | 0.004982382813760066 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.005 | P4_shadow_condensed_herding_b1 | completed_streaming | 0.251009560835953 | 0.17881930538716506 | 0.004982382813760066 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.005 | P5_shadow_condensed_hybrid_b1 | completed_streaming | 0.25413550549887015 | 0.1536286239572068 | 0.004981566163569317 | not_promoted | streaming SFT coreset over memmap block signatures |
| 0.005 | P6_best_shadow_b2_ablation | completed_derived_ablation | 0.2740235263710349 | 0.179915990105431 | 0.004982382813760066 | not_promoted | derived ablation rows are not promoted |
| 0.005 | P7_best_shadow_ks_ablation | completed_derived_ablation | 0.2740235263710349 | 0.179915990105431 | 0.004982382813760066 | not_promoted | derived ablation rows are not promoted |

- CSV: `experiments\tables\t24_products_sft_recovery_seed42.csv`
