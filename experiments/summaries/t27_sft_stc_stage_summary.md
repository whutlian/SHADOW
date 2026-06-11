# T27 SFT-STC Stage Summary

## Files Changed

- `docs/superpowers/plans/2026-06-11-t27-sft-stc.md`
- `shadow_hgc/sft/stc.py`
- `shadow_hgc/sft/stc_contract.py`
- `shadow_hgc/sft/stc_init.py`
- `shadow_hgc/sft/stc_losses.py`
- `shadow_hgc/sft/stc_trainer.py`
- `shadow_hgc/sft/timeaware_arxiv.py`
- `scripts/run_t27_stc_products.py`
- `scripts/run_t27_stc_reddit.py`
- `scripts/run_t27_arxiv_teacher_pivot.py`
- `scripts/run_t27_stage.py`
- `tests/test_t27_stc_core.py`
- `tests/test_t27_scripts.py`

## Method Names And Flags

- New main family: `sft_stc_frozen_init`, `sft_stc_trainable_delta`, `sft_stc_gradient_matching`, `sft_stc_outer_loop`, `sft_stc_outer_loop_plus_coverage`, `sft_stc_gm_plus_coverage`.
- Structure-free accounting: `ratio_mode=full_node`, `target_prototypes=syn_rows`, `shadow_nodes=0`, `condensed_edges=0`.
- Forbidden promoted flags: `uses_logits_as_input`, `uses_teacher_logits`, `uses_kd`, `uses_dense_p2`, `uses_e_by_d_materialization`, `uses_full_edge_index_on_gpu`, `uses_valid_labels`, `uses_test_labels`.
- T25/T26 HNR/FDM methods demoted to diagnostic: sft_hnr_random, sft_hnr_fdm_herding, sft_hnr_fdm_kcenter, sft_hnr_fdm_hybrid, sft_hnr_fdm_shadow_b1, sft_hnr_fdm_shadow_b2.

## Tests

- Verification result: `full pytest: 357 passed in 73.79s; latest T27 focused tests: 14 passed in 2.73s; Products long rows=18; Products tiny-ratio rows=27; Reddit long rows=80; Arxiv long/reference rows=7`
- Added tests: `tests/test_t27_stc_core.py`, `tests/test_t27_scripts.py`.

## Requirement Checklist

| requirement_check | requirement_status | notes |
|---|---|---|
| t27_schema | completed | Every generated row is written with the T27 required field list. |
| stc_structure_free_ratio | completed | Rows use ratio_mode=full_node with shadow_nodes=0 and condensed_edges=0. |
| hnr_fdm_demoted | completed | T25/T26 HNR/FDM methods are diagnostic/non-main and not promoted by default. |
| forbidden_promoted_flags | completed | No promoted row may use logits, KD, dense P2, E-by-d, full edge GPU, valid labels, or test labels. |
| products_required_rows | completed | Products required STC method grid is present for 0.25% and 0.50%. |
| reddit_required_rows | completed | Reddit required STC method grid is present for 0.50% and 1.00%. |
| arxiv_teacher_pivot_rows | completed | Arxiv teacher-pivot rows are present and condensation remains gate-controlled. |
| no_fabricated_full_results | completed | Rows without real metrics are not promoted; current Products/Reddit rows are completed_long and Arxiv is gate-blocked/reference-only. |
| performance_regression_guard | completed | No T27 row is promoted below dataset gates; smoke rows are explicitly not promoted. |

## Experiments And Outputs

| dataset | method | requested_full_node_ratio | seed | status | accuracy | macro_f1 | predicted_classes | promotion_status | failure_reason | source_table |
|---|---|---|---|---|---|---|---|---|---|---|
| ogbn-products | products_uca_mixup_frozen | 0.0025 | 42 | completed_long | 0.7361839165221855 | 0.3775724555487502 | 30 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_trainable_delta_rho005 | 0.0025 | 42 | completed_long | 0.7221284619566027 | 0.36962832228261266 | 30 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_trainable_delta_rho010 | 0.0025 | 42 | completed_long | 0.7350249040821186 | 0.3719964359045289 | 30 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_gm | 0.0025 | 42 | completed_long | 0.7331628026140814 | 0.3730035801282592 | 30 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer | 0.0025 | 42 | completed_long | 0.7350249040821186 | 0.3719964359045289 | 30 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer_plus_coverage_official | 0.0025 | 42 | completed_long | 0.7350249040821186 | 0.3719964359045289 | 30 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer_plus_coverage_balanced | 0.0025 | 42 | completed_long | 0.7145616696285874 | 0.37089446434028056 | 30 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_random_trainable_delta | 0.0025 | 42 | completed_long | 0.6918124921207488 | 0.3680959504353429 | 31 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_cb_random_trainable_delta | 0.0025 | 42 | completed_long | 0.6741706509131347 | 0.3640983408959533 | 42 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_frozen | 0.005 | 42 | completed_long | 0.7441840394272083 | 0.38447309123077245 | 31 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_trainable_delta_rho005 | 0.005 | 42 | completed_long | 0.7306450570717608 | 0.380869895423813 | 31 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_trainable_delta_rho010 | 0.005 | 42 | completed_long | 0.7387563367254216 | 0.38412119557826163 | 31 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_gm | 0.005 | 42 | completed_long | 0.7374188408881515 | 0.37902689530194533 | 31 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer | 0.005 | 42 | completed_long | 0.7387563367254216 | 0.38412119557826163 | 31 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer_plus_coverage_official | 0.005 | 42 | completed_long | 0.7387563367254216 | 0.38412119557826163 | 31 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_uca_mixup_outer_plus_coverage_balanced | 0.005 | 42 | completed_long | 0.7316160971238869 | 0.38994591805014733 | 31 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_random_trainable_delta | 0.005 | 42 | completed_long | 0.7067165335722752 | 0.3675376386275047 | 32 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| ogbn-products | products_cb_random_trainable_delta | 0.005 | 42 | completed_long | 0.7078497901803406 | 0.3774670004960359 | 42 | not_promoted | products_gate_not_met | experiments/tables/t27_stc_products_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.005 | 1 | completed_long | 0.9189630720068938 | 0.8798013875245458 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.005 | 1 | completed_long | 0.9129849379746153 | 0.8748384117640366 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.005 | 1 | completed_long | 0.9106511318959482 | 0.8725473554905528 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.005 | 1 | completed_long | 0.7265497369980073 | 0.7751350941638244 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.005 | 1 | completed_long | 0.9106511318959482 | 0.8725473554905528 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.005 | 1 | completed_long | 0.7265497369980073 | 0.7751350941638244 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.005 | 1 | completed_long | 0.8504209827118827 | 0.779558503217083 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.005 | 1 | completed_long | 0.602750300701937 | 0.6755426162425948 | 36 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.01 | 1 | completed_long | 0.9228946376317254 | 0.8847112208157808 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.01 | 1 | completed_long | 0.9178679783853653 | 0.8806014264183579 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.01 | 1 | completed_long | 0.9187835484623809 | 0.8794283149507377 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.01 | 1 | completed_long | 0.834138197224566 | 0.8249380169159245 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.01 | 1 | completed_long | 0.9187835484623809 | 0.8794283149507377 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.01 | 1 | completed_long | 0.834138197224566 | 0.8249380169159245 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.01 | 1 | completed_long | 0.8828788395598083 | 0.8095126190668086 | 40 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.01 | 1 | completed_long | 0.6023553489040088 | 0.672523216281008 | 37 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.005 | 2 | completed_long | 0.9197170708938477 | 0.8788970901229117 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.005 | 2 | completed_long | 0.918478358436709 | 0.8791404803184751 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.005 | 2 | completed_long | 0.9135414609626052 | 0.877473531365658 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.005 | 2 | completed_long | 0.7132649947040555 | 0.7651438753924354 | 40 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.005 | 2 | completed_long | 0.9135414609626052 | 0.877473531365658 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.005 | 2 | completed_long | 0.7132649947040555 | 0.7651438753924354 | 40 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.005 | 2 | completed_long | 0.8909933037717896 | 0.8330137988579273 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.005 | 2 | completed_long | 0.6318510672674721 | 0.689746840111585 | 37 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.01 | 2 | completed_long | 0.9258388237617363 | 0.8890437644641112 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.01 | 2 | completed_long | 0.9162163617758469 | 0.8805340637665989 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.01 | 2 | completed_long | 0.9185681202089654 | 0.8812995963673607 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.01 | 2 | completed_long | 0.7352027718435272 | 0.7691778951173657 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.01 | 2 | completed_long | 0.9185681202089654 | 0.8812995963673607 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.01 | 2 | completed_long | 0.7352027718435272 | 0.7691778951173657 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.01 | 2 | completed_long | 0.8978870078810836 | 0.8545902506800792 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.01 | 2 | completed_long | 0.6410426727465307 | 0.7168038953376803 | 38 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.005 | 3 | completed_long | 0.9200222609195196 | 0.8786015615667606 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.005 | 3 | completed_long | 0.9117282731630253 | 0.8699001276329164 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.005 | 3 | completed_long | 0.9090713247042349 | 0.8674065458500089 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.005 | 3 | completed_long | 0.9021058111771358 | 0.8619168990617209 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.005 | 3 | completed_long | 0.9090713247042349 | 0.8674065458500089 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.005 | 3 | completed_long | 0.9021058111771358 | 0.8619168990617209 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.005 | 3 | completed_long | 0.8965585336516885 | 0.8262853722523024 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.005 | 3 | completed_long | 0.6936789759977021 | 0.7484955362445866 | 39 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.01 | 3 | completed_long | 0.9221585910992227 | 0.8812603678487623 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.01 | 3 | completed_long | 0.9139005080516309 | 0.8734671042824504 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.01 | 3 | completed_long | 0.9091251817675888 | 0.8687423320797002 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.01 | 3 | completed_long | 0.7433890454733139 | 0.8088439106758728 | 40 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.01 | 3 | completed_long | 0.9091251817675888 | 0.8687423320797002 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.01 | 3 | completed_long | 0.7433890454733139 | 0.8088439106758728 | 40 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.01 | 3 | completed_long | 0.8082329497513598 | 0.7989910630148803 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.01 | 3 | completed_long | 0.702511534387735 | 0.7435996520756394 | 40 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.005 | 4 | completed_long | 0.9159291241046263 | 0.8761905760445775 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.005 | 4 | completed_long | 0.9095919429833222 | 0.876146049651398 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.005 | 4 | completed_long | 0.909950990072348 | 0.8768689395896132 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.005 | 4 | completed_long | 0.6733569107588461 | 0.7380780644550332 | 39 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.005 | 4 | completed_long | 0.909950990072348 | 0.8768689395896132 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.005 | 4 | completed_long | 0.6733569107588461 | 0.7380780644550332 | 39 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.005 | 4 | completed_long | 0.8535626447408577 | 0.7818006917706416 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.005 | 4 | completed_long | 0.6217259393569466 | 0.7055506355558518 | 39 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.01 | 4 | completed_long | 0.9185501678545142 | 0.8791969469776238 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.01 | 4 | completed_long | 0.9084609446528912 | 0.8741770282685968 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.01 | 4 | completed_long | 0.9021596682404898 | 0.8686434016753077 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.01 | 4 | completed_long | 0.6814175179074735 | 0.7413093200301367 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.01 | 4 | completed_long | 0.9021596682404898 | 0.8686434016753077 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.01 | 4 | completed_long | 0.6814175179074735 | 0.7413093200301367 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.01 | 4 | completed_long | 0.8553758325404377 | 0.7887560416136743 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.01 | 4 | completed_long | 0.6216002728757877 | 0.7167849112169892 | 39 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.005 | 5 | completed_long | 0.9206505933253146 | 0.8845996719158766 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.005 | 5 | completed_long | 0.8909933037717896 | 0.8628292163627247 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.005 | 5 | completed_long | 0.8695581925569539 | 0.8583061488363418 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.005 | 5 | completed_long | 0.8080354738523957 | 0.8062252513657666 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.005 | 5 | completed_long | 0.8695581925569539 | 0.8583061488363418 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.005 | 5 | completed_long | 0.8080354738523957 | 0.8062252513657666 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.005 | 5 | completed_long | 0.7708022907204279 | 0.7502144227799334 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.005 | 5 | completed_long | 0.6710410570346301 | 0.7032745614554362 | 39 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_frozen_init | 0.01 | 5 | completed_long | 0.9210814498321455 | 0.8842670126221986 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho005 | 0.01 | 5 | completed_long | 0.8968637236773603 | 0.865794780756936 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_trainable_delta_rho010 | 0.01 | 5 | completed_long | 0.8898443530869073 | 0.864409363646771 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm | 0.01 | 5 | completed_long | 0.6282426440227635 | 0.702151371015984 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_outer | 0.01 | 5 | completed_long | 0.8898443530869073 | 0.864409363646771 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_random_gm_plus_moment | 0.01 | 5 | completed_long | 0.6282426440227635 | 0.702151371015984 | 41 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_kcenter_trainable_delta | 0.01 | 5 | completed_long | 0.7429402366120317 | 0.7284881695733094 | 39 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| Reddit | reddit_medoid_trainable_delta | 0.01 | 5 | completed_long | 0.6766062869145288 | 0.7051587737528574 | 39 | not_promoted | reddit_gate_not_met | experiments/tables/t27_stc_reddit_seed42.csv |
| ogbn-arxiv | arxiv_timeaware_sft_v5_h512 | 0.0 | 42 | completed_long_reference | 0.706828796576343 | 0.5045803241909133 | 39 | not_promoted | arxiv_teacher_below_0.715 | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_timeaware_sft_v5_h768 | 0.0 | 42 | blocked_by_teacher_gate |  |  |  | not_promoted | arxiv_teacher_below_0.715 | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_timeaware_sft_v5_decay_gamma005 | 0.0 | 42 | blocked_by_teacher_gate |  |  |  | not_promoted | arxiv_teacher_below_0.715 | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_timeaware_sft_v5_decay_gamma010 | 0.0 | 42 | blocked_by_teacher_gate |  |  |  | not_promoted | arxiv_teacher_below_0.715 | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_correct_smooth_no_logits | 0.0 | 42 | blocked_by_teacher_gate |  |  |  | not_promoted | arxiv_teacher_below_0.715 | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_gnn_teacher_upper_bound | 0.0 | 42 | blocked_by_teacher_gate |  |  |  | upper_bound_diagnostic | upper_bound_diagnostic_not_promoted | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| ogbn-arxiv | arxiv_t26_best_teacher_reference | 0.0 | 42 | completed_long_reference | 0.706828796576343 | 0.5045803241909133 | 39 | not_promoted | arxiv_teacher_below_0.715 | experiments/tables/t27_arxiv_teacher_pivot_seed42.csv |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |
| stage | requirement_check |  | 42 | completed |  |  |  | not_promoted |  |  |

## Products Tiny-Ratio Sweep

Additional ogbn-products T27 SFT-STC long sweep at full-node ratios 0.02%, 0.04%, and 0.08%.

| ratio_percent | requested_full_node_ratio | best_method | syn_rows | accuracy | macro_f1 | predicted_classes | promotion_status |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.0002 | products_uca_mixup_trainable_delta_rho010 | 490 | 0.6858000868 | 0.3094500395 | 22 | not_promoted |
| 0.04 | 0.0004 | products_uca_mixup_frozen | 980 | 0.7000873439 | 0.3283601128 | 27 | not_promoted |
| 0.08 | 0.0008 | products_uca_mixup_frozen | 1959 | 0.7204511699 | 0.3483658099 | 27 | not_promoted |

- Full 27-row CSV: `experiments/tables/t27_stc_products_0p02_0p04_0p08_percent_seed42.csv`
- All rows are `completed_long`, with `promotion_status=not_promoted` and `failure_reason=products_gate_not_met`.

## Promotion Decision

- Promoted rows: `0`.
- Forbidden promoted rows: `0`.
- T27 Products and Reddit long rows completed, but no STC row met the dataset promotion gates.
- Arxiv STC remains blocked until teacher A1 accuracy >= 0.715.

## CSV Paths

- `experiments/tables/t27_stc_products_seed42.csv`
- `experiments/tables/t27_stc_products_0p02_0p04_0p08_percent_seed42.csv`
- `experiments/tables/t27_stc_reddit_seed42.csv`
- `experiments/tables/t27_arxiv_teacher_pivot_seed42.csv`
- `experiments/tables/t27_stage_summary_seed42.csv`

## Next Server Commands

```powershell
python scripts/run_t27_stc_products.py --device cuda --ratios 0.0025 0.005 --init products_uca_hybrid_mixup --methods frozen_init trainable_delta gradient_matching outer_loop outer_loop_plus_coverage --products-coverage-track official balanced --delta-rhos 0.05 0.10 0.20 --stc-outer-steps 1000 --run-long --seed 42
python scripts/run_t27_stc_reddit.py --device cuda --ratios 0.005 0.01 --init current_sft_signature_random --methods frozen_init trainable_delta gradient_matching outer_loop --delta-rhos 0.05 0.10 --run-long --seeds 1 2 3 4 5
python scripts/run_t27_arxiv_teacher_pivot.py --device cuda --variants year_features temporal_decay temporal_decay_year residual_no_logits --hidden-dims 512 768 --temporal-decay-gammas 0.05 0.10 --run-long --seed 42
python scripts/run_t27_stage.py
```

- Stage CSV: `experiments\tables\t27_stage_summary_seed42.csv`
