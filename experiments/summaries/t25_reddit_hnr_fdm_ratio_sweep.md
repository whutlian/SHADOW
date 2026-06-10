# T25 Reddit HNR-FDM-lite

- Train mode: `True`
- HNR enabled: `True`
- Rows use full-node ratio accounting and remain not_promoted unless the no-regression and T25 gates pass.

| requested_full_node_ratio | method | status | actual_full_node_ratio | accuracy | macro_f1 | predicted_classes | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|---|
| 0.001 | current_sft_signature_random | completed_streaming | 0.0010001502371600884 | 0.8983896738057197 | 0.8433886102863218 | 41 | not_promoted | acceptance_gate_not_met |
| 0.001 | current_sft_signature_medoid | completed_streaming | 0.0010001502371600884 | 0.9127515573667486 | 0.8714283868911621 | 41 | not_promoted | acceptance_gate_not_met |
| 0.001 | current_sft_signature_kcenter | completed_streaming | 0.0010001502371600884 | 0.8673680053138969 | 0.8233612044131783 | 41 | not_promoted | acceptance_gate_not_met |
| 0.001 | current_sft_signature_shadow_b1 | completed_streaming_diagnostic | 0.0010001502371600884 | 0.9153905534710878 | 0.8746059467852552 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.001 | sft_hnr_random | completed_streaming | 0.0010001502371600884 | 0.8959302012458934 | 0.8454848241770149 | 41 | not_promoted | acceptance_gate_not_met |
| 0.001 | sft_hnr_fdm_herding | completed_streaming | 0.0010001502371600884 | 0.9117103208085741 | 0.8713230962144167 | 41 | not_promoted | acceptance_gate_not_met |
| 0.001 | sft_hnr_fdm_kcenter | completed_streaming | 0.0010001502371600884 | 0.9110101789849738 | 0.8723440088839591 | 41 | not_promoted | acceptance_gate_not_met |
| 0.001 | sft_hnr_fdm_hybrid | completed_streaming | 0.0010001502371600884 | 0.9215841157567815 | 0.884890777869315 | 41 | not_promoted | acceptance_gate_not_met |
| 0.001 | sft_hnr_fdm_shadow_b1 | completed_streaming_diagnostic | 0.0010001502371600884 | 0.914780173419744 | 0.87557238881603 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.001 | sft_hnr_fdm_shadow_b2 | completed_streaming_diagnostic | 0.0010001502371600884 | 0.911333321365097 | 0.8705033164012371 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.0025 | current_sft_signature_random | completed_streaming | 0.0024982293477561006 | 0.9163958853203598 | 0.8805489216328221 | 41 | not_promoted | acceptance_gate_not_met |
| 0.0025 | current_sft_signature_medoid | completed_streaming | 0.0024982293477561006 | 0.9099868947812506 | 0.8773157884466924 | 41 | not_promoted | acceptance_gate_not_met |
| 0.0025 | current_sft_signature_kcenter | completed_streaming | 0.0024982293477561006 | 0.8960379153726011 | 0.8423379438366947 | 41 | not_promoted | acceptance_gate_not_met |
| 0.0025 | current_sft_signature_shadow_b1 | completed_streaming_diagnostic | 0.0024982293477561006 | 0.9212968780855609 | 0.8824121238133515 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.0025 | sft_hnr_random | completed_streaming | 0.0024982293477561006 | 0.9180654542843294 | 0.8795818615448815 | 41 | not_promoted | acceptance_gate_not_met |
| 0.0025 | sft_hnr_fdm_herding | completed_streaming | 0.0024982293477561006 | 0.9179577401576217 | 0.8764673025046401 | 41 | not_promoted | acceptance_gate_not_met |
| 0.0025 | sft_hnr_fdm_kcenter | completed_streaming | 0.0024982293477561006 | 0.9113512737195483 | 0.8742284765588461 | 41 | not_promoted | acceptance_gate_not_met |
| 0.0025 | sft_hnr_fdm_hybrid | completed_streaming | 0.0024982293477561006 | 0.9140441268872412 | 0.8730428739959687 | 41 | not_promoted | acceptance_gate_not_met |
| 0.0025 | sft_hnr_fdm_shadow_b1 | completed_streaming_diagnostic | 0.0024982293477561006 | 0.9127695097212 | 0.8767842467500291 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.0025 | sft_hnr_fdm_shadow_b2 | completed_streaming_diagnostic | 0.0024982293477561006 | 0.9209019262876327 | 0.8860039685317115 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.005 | current_sft_signature_random | completed_streaming | 0.005000751185800442 | 0.9233254941385562 | 0.885149317831537 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_medoid | completed_streaming | 0.005000751185800442 | 0.9228228282139203 | 0.8831134685348982 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_kcenter | completed_streaming | 0.005000751185800442 | 0.9084788970073425 | 0.8472429744875686 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | current_sft_signature_shadow_b1 | completed_streaming_diagnostic | 0.005000751185800442 | 0.9153546487621852 | 0.8774389012986487 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.005 | sft_hnr_random | completed_streaming | 0.005000751185800442 | 0.9170601224350574 | 0.8803582097707346 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | sft_hnr_fdm_herding | completed_streaming | 0.005000751185800442 | 0.9206685456797659 | 0.8863924816220402 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | sft_hnr_fdm_kcenter | completed_streaming | 0.005000751185800442 | 0.9231280182395921 | 0.882975902096398 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | sft_hnr_fdm_hybrid | completed_streaming | 0.005000751185800442 | 0.9217097822379405 | 0.8817167425644433 | 41 | not_promoted | no_regression_gate_not_met |
| 0.005 | sft_hnr_fdm_shadow_b1 | completed_streaming_diagnostic | 0.005000751185800442 | 0.9197529756027503 | 0.8794423975650453 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.005 | sft_hnr_fdm_shadow_b2 | completed_streaming_diagnostic | 0.005000751185800442 | 0.9196452614760425 | 0.8798299763214216 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.01 | current_sft_signature_random | completed_streaming | 0.010001502371600884 | 0.9233793512019102 | 0.886217047127179 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_medoid | completed_streaming | 0.010001502371600884 | 0.9222663052259303 | 0.8814878728371921 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_kcenter | completed_streaming | 0.010001502371600884 | 0.9219431628458072 | 0.8801208622952813 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | current_sft_signature_shadow_b1 | completed_streaming_diagnostic | 0.010001502371600884 | 0.920237689172935 | 0.8745109862626925 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.01 | sft_hnr_random | completed_streaming | 0.010001502371600884 | 0.9205787839075095 | 0.8852204113072069 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | sft_hnr_fdm_herding | completed_streaming | 0.010001502371600884 | 0.9127695097212 | 0.87526898568368 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | sft_hnr_fdm_kcenter | completed_streaming | 0.010001502371600884 | 0.923558874746423 | 0.8865678667855177 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | sft_hnr_fdm_hybrid | completed_streaming | 0.010001502371600884 | 0.9236127318097769 | 0.8881558607412497 | 41 | not_promoted | acceptance_gate_not_met |
| 0.01 | sft_hnr_fdm_shadow_b1 | completed_streaming_diagnostic | 0.010001502371600884 | 0.924564206595695 | 0.8881542321172001 | 41 | not_promoted | shadow_materialization_not_trained |
| 0.01 | sft_hnr_fdm_shadow_b2 | completed_streaming_diagnostic | 0.010001502371600884 | 0.9180654542843294 | 0.8800353437565743 | 41 | not_promoted | shadow_materialization_not_trained |

- CSV: `experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv`
