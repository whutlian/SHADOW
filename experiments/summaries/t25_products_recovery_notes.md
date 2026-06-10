# T25 Products Recovery Ladder

- Train mode: `True`
- Rows are diagnostic unless the explicit products recovery gates pass.

| requested_full_node_ratio | ladder_stage | method | status | accuracy | macro_f1 | predicted_classes | actual_full_node_ratio | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|---|---|
| 0.0005 | P0 | P0_identity_replay | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | 47 | 0.0005001982418338044 | not_promoted | identity_reference_only |
| 0.0005 | P1 | P1_selected_real_prototypes_replay | completed_streaming | 0.10049609347288475 | 0.05600118653478825 | 38 | 0.0005001982418338044 | not_promoted | products_t25_gate_not_met |
| 0.0005 | P2 | P2_hnr_fdm_prototype_oracle | completed_streaming | 0.0006931481805312118 | 0.005752349819509904 | 29 | 0.0005001982418338044 | not_promoted | products_t25_gate_not_met |
| 0.0005 | P3 | P3_hnr_fdm_shadow_b1 | completed_streaming | 0.007570859038331456 | 0.02498478202578895 | 38 | 0.0005001982418338044 | not_promoted | shadow_materialization_not_trained |
| 0.0005 | P3 | P3_hnr_fdm_shadow_b2 | completed_streaming | 0.0286151812103524 | 0.037082679156536756 | 40 | 0.0005001982418338044 | not_promoted | shadow_materialization_not_trained |
| 0.001 | P0 | P0_identity_replay | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | 47 | 0.0009999881585722342 | not_promoted | identity_reference_only |
| 0.001 | P1 | P1_selected_real_prototypes_replay | completed_streaming | 0.06170690676524372 | 0.0571249905020672 | 38 | 0.0009999881585722342 | not_promoted | products_t25_gate_not_met |
| 0.001 | P2 | P2_hnr_fdm_prototype_oracle | completed_streaming | 0.04874087870765369 | 0.0313827694162128 | 37 | 0.0009999881585722342 | not_promoted | products_t25_gate_not_met |
| 0.001 | P3 | P3_hnr_fdm_shadow_b1 | completed_streaming | 0.03653894033277438 | 0.03821091279641932 | 37 | 0.0009999881585722342 | not_promoted | shadow_materialization_not_trained |
| 0.001 | P3 | P3_hnr_fdm_shadow_b2 | completed_streaming | 0.03389467491395519 | 0.03317839398667312 | 33 | 0.0009999881585722342 | not_promoted | shadow_materialization_not_trained |
| 0.0025 | P0 | P0_identity_replay | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | 47 | 0.0025001745589782725 | not_promoted | identity_reference_only |
| 0.0025 | P1 | P1_selected_real_prototypes_replay | completed_streaming | 0.08198804296795749 | 0.04286723753284472 | 34 | 0.002489966431593909 | not_promoted | products_t25_gate_not_met |
| 0.0025 | P2 | P2_hnr_fdm_prototype_oracle | completed_streaming | 0.1261082350432043 | 0.09777620150207107 | 36 | 0.0025001745589782725 | not_promoted | products_t25_gate_not_met |
| 0.0025 | P3 | P3_hnr_fdm_shadow_b1 | completed_streaming | 0.2046097517002238 | 0.12613426122985633 | 31 | 0.0025001745589782725 | not_promoted | shadow_materialization_not_trained |
| 0.0025 | P3 | P3_hnr_fdm_shadow_b2 | completed_streaming | 0.16920632725902368 | 0.1183319577852288 | 36 | 0.0025001745589782725 | not_promoted | shadow_materialization_not_trained |
| 0.005 | P0 | P0_identity_replay | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | 47 | 0.00499994079286117 | not_promoted | identity_reference_only |
| 0.005 | P1 | P1_selected_real_prototypes_replay | completed_streaming | 0.1519146749952894 | 0.0967926371510277 | 30 | 0.004959924933514466 | not_promoted | products_t25_gate_not_met |
| 0.005 | P2 | P2_hnr_fdm_prototype_oracle | completed_streaming | 0.28839031020414435 | 0.17021558791081592 | 37 | 0.00499994079286117 | not_promoted | products_t25_gate_not_met |
| 0.005 | P3 | P3_hnr_fdm_shadow_b1 | completed_streaming | 0.26935810592515175 | 0.20213225646425712 | 40 | 0.00499994079286117 | not_promoted | shadow_materialization_not_trained |
| 0.005 | P3 | P3_hnr_fdm_shadow_b2 | completed_streaming | 0.16941915176556227 | 0.14921068889307768 | 37 | 0.00499994079286117 | not_promoted | shadow_materialization_not_trained |

- CSV: `experiments\tables\t25_products_recovery_ladder_seed42.csv`
