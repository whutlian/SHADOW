# T0-S SFB-v2 Stage Summary

This stage keeps Shadow-HGC-R-1 frozen and implements SFB-v2 as an opt-in scalable fullgraph signal generator.

## Code Changes

- Added bounded typed feature demand, target table memmap I/O, meta-path table evaluation, SCAP-v2 sparse/top-k helpers, low-dimensional logit propagation, and structural stats.
- Added strong self encoder and `BlockGatedTableModel` with train-target-row block stats, residual branch gates, and raw logits.
- Added SFB-v2 fullgraph, scalability, condensation recovery, and stage runner scripts.

## Best Fullgraph Rows

| dataset | variant | status | accuracy | macro_f1 | weighted_f1 | gate_acc | gate_acc_passed | recovery_gate_passed | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | B3_scap_v2 | completed | 0.9154863357543945 | 0.9165802995363871 | 0.9157890029310037 | 0.93 | False | False | completed |
| dblp | B0_self | completed | 0.7700704336166382 | 0.7617398649454117 | 0.7696313615510597 | 0.91 | False | False | completed |
| imdb | B2_metapath | completed | 0.35352903604507446 | 0.2843838885426521 | 0.335682670482627 | 0.6 | False | False | completed |
| ogbn-arxiv | B3_scap_v2 | completed | 0.516819953918457 | 0.1844054448418319 | 0.46646713416923796 | 0.7 | False | False | completed_bounded_edges_5000000 |
| ogbn-products | B4_logit_prop | completed | 0.17556576430797577 | 0.05002565026138364 | 0.18192531351689956 | 0.72 | False | False | completed_bounded_edges_5000000 |

## Block Effects

| dataset | best_block_variant | best_acc | worst_block_variant | worst_acc | deltas_vs_B0 |
| --- | --- | --- | --- | --- | --- |
| acm | B3_scap_v2 | 0.9154863357543945 | B0_self | 0.8418319225311279 | {'B0_self': 0.0, 'B1_typed_demand': 0.072238, 'B2_metapath': 0.062795, 'B3_scap_v2': 0.073654, 'B4_logit_prop': 0.067044} |
| dblp | B0_self | 0.7700704336166382 | B1_typed_demand | 0.5095070600509644 | {'B0_self': 0.0, 'B1_typed_demand': -0.260563, 'B2_metapath': -0.134507, 'B3_scap_v2': -0.134507, 'B4_logit_prop': -0.134507} |
| imdb | B2_metapath | 0.35352903604507446 | B0_self | 0.3332292437553406 | {'B0_self': 0.0, 'B1_typed_demand': 0.014366, 'B2_metapath': 0.0203, 'B3_scap_v2': 0.0203, 'B4_logit_prop': 0.0203} |
| ogbn-arxiv | B3_scap_v2 | 0.516819953918457 | B0_self | 0.35767340660095215 | {'B0_self': 0.0, 'B1_typed_demand': 0.157274, 'B2_metapath': 0.157274, 'B3_scap_v2': 0.159147, 'B4_logit_prop': 0.120754} |
| ogbn-products | B4_logit_prop | 0.17556576430797577 | B3_scap_v2 | 0.013050977140665054 | {'B0_self': 0.0, 'B1_typed_demand': -0.020916, 'B2_metapath': -0.020916, 'B3_scap_v2': -0.021447, 'B4_logit_prop': 0.141068} |

## Full Ablation Rows

| dataset | variant | status | accuracy | macro_f1 | weighted_f1 | enabled_blocks | medium_execution_mode | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | B0_self | completed | 0.8418319225311279 | 0.8419868350028992 | 0.8411634734487047 | ["self"] | small_full_schema | completed |
| acm | B1_typed_demand | completed | 0.9140698909759521 | 0.9146179556846619 | 0.9136718249593543 | ["self", "typed:cite_ref", "typed:cited_by", "typed:writes", "typed:subject_of"] | small_full_schema | completed |
| acm | B2_metapath | completed | 0.9046270251274109 | 0.9060052831967672 | 0.905125766884685 | ["self", "typed:cite_ref", "typed:cited_by", "typed:writes", "typed:subject_of", "metapath:PAP", "metapath:PSP", "metapath:PTP"] | small_full_schema | completed |
| acm | B3_scap_v2 | completed | 0.9154863357543945 | 0.9165802995363871 | 0.9157890029310037 | ["self", "typed:cite_ref", "typed:cited_by", "typed:writes", "typed:subject_of", "metapath:PAP", "metapath:PSP", "metapath:PTP", "scap_v2:cite_ref", "scap_v2:cited_by"] | small_full_schema | completed |
| acm | B4_logit_prop | completed | 0.9088762998580933 | 0.909429669380188 | 0.9087237532123095 | ["self", "typed:cite_ref", "typed:cited_by", "typed:writes", "typed:subject_of", "metapath:PAP", "metapath:PSP", "metapath:PTP", "scap_v2:cite_ref", "scap_v2:cited_by", "logit_prop"] | small_full_schema | completed |
| dblp | B0_self | completed | 0.7700704336166382 | 0.7617398649454117 | 0.7696313615510597 | ["self"] | small_full_schema | completed |
| dblp | B1_typed_demand | completed | 0.5095070600509644 | 0.4975796565413475 | 0.5044004215407787 | ["self", "typed:written_by"] | small_full_schema | completed |
| dblp | B2_metapath | completed | 0.6355633735656738 | 0.6332688480615616 | 0.6334858400243727 | ["self", "typed:written_by", "metapath:APA", "metapath:APVPA", "metapath:APTPA"] | small_full_schema | completed |
| dblp | B3_scap_v2 | completed | 0.6355633735656738 | 0.6332688480615616 | 0.6334858400243727 | ["self", "typed:written_by", "metapath:APA", "metapath:APVPA", "metapath:APTPA"] | small_full_schema | completed |
| dblp | B4_logit_prop | completed | 0.6355633735656738 | 0.6332688480615616 | 0.6334858400243727 | ["self", "typed:written_by", "metapath:APA", "metapath:APVPA", "metapath:APTPA"] | small_full_schema | completed |
| dblp | DBLP_APA_only | diagnostic_existing | 0.6355633735656738 | 0.6332688480615616 | 0.6334858400243727 | ["self", "typed:written_by", "metapath:APA", "metapath:APVPA", "metapath:APTPA"] | small_full_schema | completed |
| dblp | DBLP_APA_APVPA | diagnostic_existing | 0.6355633735656738 | 0.6332688480615616 | 0.6334858400243727 | ["self", "typed:written_by", "metapath:APA", "metapath:APVPA", "metapath:APTPA"] | small_full_schema | completed |
| dblp | DBLP_APA_APVPA_APTPA | diagnostic_existing | 0.6355633735656738 | 0.6332688480615616 | 0.6334858400243727 | ["self", "typed:written_by", "metapath:APA", "metapath:APVPA", "metapath:APTPA"] | small_full_schema | completed |
| dblp | DBLP_APA_APVPA_APTPA_APCPA | diagnostic_existing | 0.6355633735656738 | 0.6332688480615616 | 0.6334858400243727 | ["self", "typed:written_by", "metapath:APA", "metapath:APVPA", "metapath:APTPA"] | small_full_schema | APCPA schema_missing in local DBLP full schema |
| imdb | B0_self | completed | 0.3332292437553406 | 0.2967915952205658 | 0.3260893605530315 | ["self"] | small_full_schema | completed |
| imdb | B1_typed_demand | completed | 0.3475952446460724 | 0.2827941611409187 | 0.32805823537319 | ["self", "typed:directs", "typed:acts_in"] | small_full_schema | completed |
| imdb | B2_metapath | completed | 0.35352903604507446 | 0.2843838885426521 | 0.335682670482627 | ["self", "typed:directs", "typed:acts_in", "metapath:MAM", "metapath:MDM", "metapath:MKM"] | small_full_schema | completed |
| imdb | B3_scap_v2 | completed | 0.35352903604507446 | 0.2843838885426521 | 0.335682670482627 | ["self", "typed:directs", "typed:acts_in", "metapath:MAM", "metapath:MDM", "metapath:MKM"] | small_full_schema | completed |
| imdb | B4_logit_prop | completed | 0.35352903604507446 | 0.2843838885426521 | 0.335682670482627 | ["self", "typed:directs", "typed:acts_in", "metapath:MAM", "metapath:MDM", "metapath:MKM"] | small_full_schema | completed |
| ogbn-arxiv | B0_self | completed | 0.35767340660095215 | 0.09558469322510063 | 0.3037463948870476 | ["self"] | bounded_edges | completed_bounded_edges_5000000 |
| ogbn-arxiv | B1_typed_demand | completed | 0.5149476528167725 | 0.19514889530837537 | 0.47608575316647916 | ["self", "typed:cite_ref", "typed:cited_by", "structure"] | bounded_edges | completed_bounded_edges_5000000 |
| ogbn-arxiv | B2_metapath | completed | 0.5149476528167725 | 0.19514889530837537 | 0.47608575316647916 | ["self", "typed:cite_ref", "typed:cited_by", "structure"] | bounded_edges | completed_bounded_edges_5000000 |
| ogbn-arxiv | B3_scap_v2 | completed | 0.516819953918457 | 0.1844054448418319 | 0.46646713416923796 | ["self", "typed:cite_ref", "typed:cited_by", "structure", "scap_v2:cite_ref", "scap_v2:cited_by"] | bounded_edges | completed_bounded_edges_5000000 |
| ogbn-arxiv | B4_logit_prop | completed | 0.4784272611141205 | 0.15381201778072864 | 0.42901234039119374 | ["self", "typed:cite_ref", "typed:cited_by", "structure", "scap_v2:cite_ref", "scap_v2:cited_by", "logit_prop"] | bounded_edges | completed_bounded_edges_5000000 |
| ogbn-products | B0_self | completed | 0.03449790179729462 | 0.02563670493178661 | 0.05507294989658261 | ["self"] | bounded_edges | completed_bounded_edges_5000000 |
| ogbn-products | B1_typed_demand | completed | 0.01358190830796957 | 0.020973218601429446 | 0.02545290968461757 | ["self", "typed:co_purchase", "typed:co_purchased_by", "structure"] | bounded_edges | completed_bounded_edges_5000000 |
| ogbn-products | B2_metapath | completed | 0.01358190830796957 | 0.020973218601429446 | 0.02545290968461757 | ["self", "typed:co_purchase", "typed:co_purchased_by", "structure"] | bounded_edges | completed_bounded_edges_5000000 |
| ogbn-products | B3_scap_v2 | completed | 0.013050977140665054 | 0.017244928872505044 | 0.02440233183470654 | ["self", "typed:co_purchase", "typed:co_purchased_by", "structure", "scap_v2:co_purchase", "scap_v2:co_purchased_by"] | bounded_edges | completed_bounded_edges_5000000 |
| ogbn-products | B4_logit_prop | completed | 0.17556576430797577 | 0.05002565026138364 | 0.18192531351689956 | ["self", "typed:co_purchase", "typed:co_purchased_by", "structure", "scap_v2:co_purchase", "scap_v2:co_purchased_by", "logit_prop"] | bounded_edges | completed_bounded_edges_5000000 |

## Scalability

| dataset | status | num_nodes | num_edges | edge_scans | disk_bytes | valid_scalability |
| --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | completed_or_bounded_in_fullgraph_table | 169343 | 2332486 | 5 | 146312352 | True |
| ogbn-products | completed_or_bounded_in_fullgraph_table | 2449029 | 123718280 | 5 | 2184533868 | True |
| ogbn-papers100M | dry_run_estimate | 111059956 | 1615685872 | 5 | 154595458752 | True |
| mag240m | dry_run_estimate | 121751666 | 1728364232 | 5 | 160225192456 | True |

## Condensation Recovery Gate

| dataset | fullgraph_variant | fullgraph_acc | recovery_row | status | promoted |
| --- | --- | --- | --- | --- | --- |
| acm | B3_scap_v2 | 0.9154863357543945 | identity_condensed | blocked_by_sfb_v2_fullgraph_gate | False |
| acm | B3_scap_v2 | 0.9154863357543945 | prototype_oracle | blocked_by_sfb_v2_fullgraph_gate | False |
| acm | B3_scap_v2 | 0.9154863357543945 | shadow_hgc_sfb_v2_signal | blocked_by_sfb_v2_fullgraph_gate | False |
| dblp | B0_self | 0.7700704336166382 | identity_condensed | blocked_by_sfb_v2_fullgraph_gate | False |
| dblp | B0_self | 0.7700704336166382 | prototype_oracle | blocked_by_sfb_v2_fullgraph_gate | False |
| dblp | B0_self | 0.7700704336166382 | shadow_hgc_sfb_v2_signal | blocked_by_sfb_v2_fullgraph_gate | False |
| imdb | B2_metapath | 0.35352903604507446 | identity_condensed | blocked_by_sfb_v2_fullgraph_gate | False |
| imdb | B2_metapath | 0.35352903604507446 | prototype_oracle | blocked_by_sfb_v2_fullgraph_gate | False |
| imdb | B2_metapath | 0.35352903604507446 | shadow_hgc_sfb_v2_signal | blocked_by_sfb_v2_fullgraph_gate | False |
| ogbn-arxiv | B3_scap_v2 | 0.516819953918457 | identity_condensed | blocked_by_sfb_v2_fullgraph_gate | False |
| ogbn-arxiv | B3_scap_v2 | 0.516819953918457 | prototype_oracle | blocked_by_sfb_v2_fullgraph_gate | False |
| ogbn-arxiv | B3_scap_v2 | 0.516819953918457 | shadow_hgc_sfb_v2_signal | blocked_by_sfb_v2_fullgraph_gate | False |
| ogbn-products | B4_logit_prop | 0.17556576430797577 | identity_condensed | blocked_by_sfb_v2_fullgraph_gate | False |
| ogbn-products | B4_logit_prop | 0.17556576430797577 | prototype_oracle | blocked_by_sfb_v2_fullgraph_gate | False |
| ogbn-products | B4_logit_prop | 0.17556576430797577 | shadow_hgc_sfb_v2_signal | blocked_by_sfb_v2_fullgraph_gate | False |

## Required Questions

- Which blocks improve/hurt each dataset: see `Block Effects`; ACM improves most with B3, DBLP is hurt by current feature/metapath blocks versus B0, IMDB improves slightly with B2, arxiv improves with B3, and products improves most with B4.
- Medium run status: arxiv/products have completed SFB-v2 rows instead of `skipped_resource_guard`; products uses bounded local edge execution (`completed_bounded_edges_5000000`) to avoid desktop OOM.
- Scalability preservation: promoted rows log `uses_diffusion=false`, `uses_dense_p2=false`, `uses_dense_metapath_adjacency=false`, `uses_full_graph_backprop=false`, and `uses_e_by_d_materialization=false`.
- Condensation eligibility: no dataset passed the full accuracy gate, so no condensation recovery row is promoted.
- Bottleneck: current blocker is fullgraph signal quality, not prototype loss or shadow factorization; recovery rows remain gate-blocked.
- Eligible datasets: `[]`.

## Medium Execution Details

| dataset | variant | status | accuracy | medium_execution_mode | reason | peak_cpu_ram_gb |
| --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | B0_self | completed | 0.35767340660095215 | bounded_edges | completed_bounded_edges_5000000 | 3.126789093017578 |
| ogbn-arxiv | B1_typed_demand | completed | 0.5149476528167725 | bounded_edges | completed_bounded_edges_5000000 | 3.1698532104492188 |
| ogbn-arxiv | B2_metapath | completed | 0.5149476528167725 | bounded_edges | completed_bounded_edges_5000000 | 3.181549072265625 |
| ogbn-arxiv | B3_scap_v2 | completed | 0.516819953918457 | bounded_edges | completed_bounded_edges_5000000 | 3.2770843505859375 |
| ogbn-arxiv | B4_logit_prop | completed | 0.4784272611141205 | bounded_edges | completed_bounded_edges_5000000 | 2.0685653686523438 |
| ogbn-products | B0_self | completed | 0.03449790179729462 | bounded_edges | completed_bounded_edges_5000000 | 7.217353820800781 |
| ogbn-products | B1_typed_demand | completed | 0.01358190830796957 | bounded_edges | completed_bounded_edges_5000000 | 10.003658294677734 |
| ogbn-products | B2_metapath | completed | 0.01358190830796957 | bounded_edges | completed_bounded_edges_5000000 | 9.991218566894531 |
| ogbn-products | B3_scap_v2 | completed | 0.013050977140665054 | bounded_edges | completed_bounded_edges_5000000 | 13.56113052368164 |
| ogbn-products | B4_logit_prop | completed | 0.17556576430797577 | bounded_edges | completed_bounded_edges_5000000 | 13.615497589111328 |

## Artifacts

- `experiments\tables\t0s_sfb_v2_fullgraph_seed42.csv`
- `experiments\tables\t0s_sfb_v2_scalability_seed42.csv`
- `experiments\tables\t0s_sfb_v2_condensation_recovery_seed42.csv`
