# Shadow-HGC-SOTA Sprint Summary

## Scope

- Seed policy: single seed `42`.
- Default R-1 path remains unchanged; SOTA features are opt-in scripts/flags.
- Diffusion is not promoted.
- Components implemented: explicit compiled block stats, meta-path feature blocks, SeHGNN-lite module, Path-LAD, coverage medoids, source anchors, teacher cache/KD.

## Best Row Per Dataset
| dataset | variant | accuracy | macro_f1 | gap_to_previous_shadow | gap_to_sota_gate | total_condensed_node_ratio | prototype_mode | teacher_type | use_kd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | S1_sehgnn_lite_metapath | 0.8729934096336365 | 0.8743000626564026 | 0.0184 | -0.0270 | 0.013068908791811369 | kmeans_mean | none | False |
| dblp | S0_current_best | 0.8056337833404541 | 0.7980579286813736 | -0.0314 | -0.1044 | 0.013433864053888548 | kmeans_mean | none | False |
| imdb | S0_current_best | 0.369768887758255 | 0.34745914340019224 | -0.0502 | -0.1802 | 0.04281045751633987 | kmeans_mean | none | False |
| ogbn-arxiv | S2_coverage_medoids | 0.4671933948993683 | 0.2052860957570374 | -0.1340 | -0.2128 | 0.005001682974790809 | coverage_medoid | none | False |
| ogbn-products | S0_current_best | 0.5135102868080139 | 0.18708413856183279 | -0.1452 | -0.1865 | 0.0005001982418338044 | kmeans_mean | none | False |

## Component Answers

1. SeHGNN-lite/meta-path: partially. ACM S1 reaches `0.8730` vs S0 `0.8234` and is `-0.0270` from the 0.90 gate; DBLP does not close the gap because best S1 is `0.4761` vs S0 `0.8056`.
2. Source anchors + Path-LAD did not rescue IMDB. Best S3 is `0.3054` and best S4 is `0.2870`, both below S0 `0.3698` and below the 0.55 gate.
3. KD did not close arxiv/products. Arxiv best S4 is `0.3736` vs S2 `0.4672`; products S4 rows were `timeout_dropped`, and the completed products S0 row is `0.5135` at 0.05% full-node ratio.
4. Biggest gain: ACM S1 at 4.8% target ratio is the clear positive result, improving over ACM S0 at the same ratio by about +0.1634 accuracy.
5. Schema preservation: yes. Meta-path and Path-LAD are feature blocks; source-anchor utilities expose original source types/relations only.
6. Compression comparability: medium rows use requested full-node ratios; small rows report total condensed node ratio, which is lower than the requested train-target ratio and must be compared using that logged field.
7. Paper positioning: SOTA mode should remain a performance branch. The training-free Lite/R-1 path remains the main scalable method; teacher/KD is not training-free and did not pass gates in this sprint.

## Small Rows
| dataset | variant | requested_ratio | accuracy | macro_f1 | predicted_class_count | total_condensed_node_ratio | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acm | S0_current_best | 0.012 | 0.6496695280075073 | 0.6117899715900421 | 3 | 0.005026503381465911 | completed |
| acm | S1_sehgnn_lite_metapath | 0.012 | 0.477337121963501 | 0.41942377388477325 | 3 | 0.005026503381465911 | completed |
| acm | S2_coverage_medoids | 0.012 | 0.580736517906189 | 0.5811920464038849 | 3 | 0.0049351124108938035 | completed |
| acm | S3_path_lad_source_anchor | 0.012 | 0.5335221886634827 | 0.5181722690661749 | 3 | 0.0049351124108938035 | completed |
| acm | S4_teacher_kd | 0.012 | 0.5250235795974731 | 0.5354994833469391 | 3 | 0.0049351124108938035 | completed |
| acm | S0_current_best | 0.024 | 0.698300302028656 | 0.6789692342281342 | 3 | 0.0071284957046243835 | completed |
| acm | S1_sehgnn_lite_metapath | 0.024 | 0.7691218256950378 | 0.7470992008845011 | 3 | 0.0071284957046243835 | completed |
| acm | S2_coverage_medoids | 0.024 | 0.6090651750564575 | 0.5882640480995178 | 3 | 0.007402668616340705 | completed |
| acm | S3_path_lad_source_anchor | 0.024 | 0.6610009670257568 | 0.6498543620109558 | 3 | 0.007402668616340705 | completed |
| acm | S4_teacher_kd | 0.024 | 0.4107648730278015 | 0.37992556889851886 | 3 | 0.007402668616340705 | completed |
| acm | S0_current_best | 0.048 | 0.7096317410469055 | 0.6963719328244528 | 3 | 0.013068908791811369 | completed |
| acm | S1_sehgnn_lite_metapath | 0.048 | 0.8729934096336365 | 0.8743000626564026 | 3 | 0.013068908791811369 | completed |
| acm | S2_coverage_medoids | 0.048 | 0.7219074368476868 | 0.7159724235534668 | 3 | 0.013160299762383476 | completed |
| acm | S3_path_lad_source_anchor | 0.048 | 0.6676109433174133 | 0.677187462647756 | 3 | 0.013160299762383476 | completed |
| acm | S4_teacher_kd | 0.048 | 0.3243626058101654 | 0.25431596239407855 | 3 | 0.013160299762383476 | completed |
| acm | S0_current_best | 0.096 | 0.8234183192253113 | 0.823662281036377 | 3 | 0.024584171083896912 | completed |
| acm | S1_sehgnn_lite_metapath | 0.096 | 0.8682719469070435 | 0.8684150377909342 | 3 | 0.024584171083896912 | completed |
| acm | S2_coverage_medoids | 0.096 | 0.7455146312713623 | 0.7218442956606547 | 3 | 0.027051727289343814 | completed |
| acm | S3_path_lad_source_anchor | 0.096 | 0.8054768443107605 | 0.8070018887519836 | 3 | 0.027051727289343814 | completed |
| acm | S4_teacher_kd | 0.096 | 0.36307838559150696 | 0.28975630179047585 | 3 | 0.027051727289343814 | completed |
| dblp | S0_current_best | 0.012 | 0.794718325138092 | 0.7870782613754272 | 4 | 0.001837109614206981 | completed |
| dblp | S1_sehgnn_lite_metapath | 0.012 | 0.459859162569046 | 0.4543410912156105 | 4 | 0.001837109614206981 | completed |
| dblp | S2_coverage_medoids | 0.012 | 0.3390845060348511 | 0.3266420140862465 | 4 | 0.0017988364972443356 | completed |
| dblp | S3_path_lad_source_anchor | 0.012 | 0.3257042169570923 | 0.2879209816455841 | 4 | 0.0017988364972443356 | completed |
| dblp | S4_teacher_kd | 0.012 | 0.26056337356567383 | 0.24460675567388535 | 4 | 0.0017988364972443356 | completed |
| dblp | S0_current_best | 0.024 | 0.7795774936676025 | 0.772317111492157 | 4 | 0.0014543784445805266 | completed |
| dblp | S1_sehgnn_lite_metapath | 0.024 | 0.3813380300998688 | 0.35873496532440186 | 4 | 0.0014543784445805266 | completed |
| dblp | S2_coverage_medoids | 0.024 | 0.38204225897789 | 0.36737966537475586 | 4 | 0.0033297611757501532 | completed |
| dblp | S3_path_lad_source_anchor | 0.024 | 0.36901408433914185 | 0.3373152054846287 | 4 | 0.0033297611757501532 | completed |
| dblp | S4_teacher_kd | 0.024 | 0.2950704097747803 | 0.27923572435975075 | 4 | 0.0033297611757501532 | completed |
| dblp | S0_current_best | 0.048 | 0.8024647831916809 | 0.7956401407718658 | 4 | 0.005319963257807716 | completed |
| dblp | S1_sehgnn_lite_metapath | 0.048 | 0.4760563373565674 | 0.49529415369033813 | 4 | 0.005319963257807716 | completed |
| dblp | S2_coverage_medoids | 0.048 | 0.3964788615703583 | 0.4042668119072914 | 4 | 0.0066595223515003065 | completed |
| dblp | S3_path_lad_source_anchor | 0.048 | 0.3795774579048157 | 0.375626876950264 | 4 | 0.0066595223515003065 | completed |
| dblp | S4_teacher_kd | 0.048 | 0.3426056206226349 | 0.31900398060679436 | 4 | 0.0066595223515003065 | completed |
| dblp | S0_current_best | 0.096 | 0.8056337833404541 | 0.7980579286813736 | 4 | 0.013433864053888548 | completed |
| dblp | S1_sehgnn_lite_metapath | 0.096 | 0.4711267650127411 | 0.4898292198777199 | 4 | 0.013433864053888548 | completed |
| dblp | S2_coverage_medoids | 0.096 | 0.4327464699745178 | 0.4322882369160652 | 4 | 0.013433864053888548 | completed |
| dblp | S3_path_lad_source_anchor | 0.096 | 0.40457746386528015 | 0.40504685789346695 | 4 | 0.013433864053888548 | completed |
| dblp | S4_teacher_kd | 0.096 | 0.32042253017425537 | 0.29411240108311176 | 4 | 0.013433864053888548 | completed |
| imdb | S0_current_best | 0.012 | 0.3425983786582947 | 0.3311773508787155 | 5 | 0.005929038281979458 | completed |
| imdb | S1_sehgnn_lite_metapath | 0.012 | 0.3194878101348877 | 0.2861614376306534 | 5 | 0.005929038281979458 | completed |
| imdb | S2_coverage_medoids | 0.012 | 0.18019987642765045 | 0.1812779501080513 | 5 | 0.006255835667600374 | completed |
| imdb | S3_path_lad_source_anchor | 0.012 | 0.30543410778045654 | 0.23099800050258637 | 5 | 0.006255835667600374 | completed |
| imdb | S4_teacher_kd | 0.012 | 0.2367270439863205 | 0.15373046025633813 | 5 | 0.006255835667600374 | completed |
| imdb | S0_current_best | 0.024 | 0.3582136034965515 | 0.337499076128006 | 5 | 0.010784313725490196 | completed |
| imdb | S1_sehgnn_lite_metapath | 0.024 | 0.3366645872592926 | 0.283083975315094 | 5 | 0.010784313725490196 | completed |
| imdb | S2_coverage_medoids | 0.024 | 0.3094940781593323 | 0.2650454223155975 | 5 | 0.010784313725490196 | completed |
| imdb | S3_path_lad_source_anchor | 0.024 | 0.2042473405599594 | 0.19030728489160537 | 5 | 0.010784313725490196 | completed |
| imdb | S4_teacher_kd | 0.024 | 0.2870081067085266 | 0.162602255679667 | 5 | 0.010784313725490196 | completed |
| imdb | S0_current_best | 0.048 | 0.34415990114212036 | 0.33370828032493594 | 5 | 0.021568627450980392 | completed |
| imdb | S1_sehgnn_lite_metapath | 0.048 | 0.3257339298725128 | 0.2710739761590958 | 5 | 0.021568627450980392 | completed |
| imdb | S2_coverage_medoids | 0.048 | 0.30387258529663086 | 0.23062760829925538 | 5 | 0.021568627450980392 | completed |
| imdb | S3_path_lad_source_anchor | 0.048 | 0.28107431530952454 | 0.23565983027219772 | 5 | 0.021568627450980392 | completed |
| imdb | S4_teacher_kd | 0.048 | 0.25515303015708923 | 0.20060989558696746 | 5 | 0.021568627450980392 | completed |
| imdb | S0_current_best | 0.096 | 0.369768887758255 | 0.34745914340019224 | 5 | 0.04281045751633987 | completed |
| imdb | S1_sehgnn_lite_metapath | 0.096 | 0.3069956302642822 | 0.2806369125843048 | 5 | 0.04281045751633987 | completed |
| imdb | S2_coverage_medoids | 0.096 | 0.29387882351875305 | 0.24864265620708464 | 5 | 0.043137254901960784 | completed |
| imdb | S3_path_lad_source_anchor | 0.096 | 0.2963772714138031 | 0.2426243707537651 | 5 | 0.043137254901960784 | completed |
| imdb | S4_teacher_kd | 0.096 | 0.2760774493217468 | 0.14215327352285384 | 5 | 0.043137254901960784 | completed |

## Medium Rows
| dataset | variant | requested_full_condensed_node_ratio | accuracy | macro_f1 | predicted_class_count | total_condensed_node_ratio | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | S0_current_best | 0.0005 | 0.3409666121006012 | 0.20036885540466756 | 40 | 0.000501939849890459 | completed |
| ogbn-arxiv | S2_coverage_medoids | 0.0005 | 0.24883237481117249 | 0.03835016243392601 | 23 | 0.000501939849890459 | completed |
| ogbn-arxiv | S4_teacher_kd | 0.0005 | 0.1873341202735901 | 0.0315606064224994 | 5 | 0.000501939849890459 | completed |
| ogbn-arxiv | S0_current_best | 0.0025 | 0.3779190480709076 | 0.22030348842963576 | 38 | 0.002497888900043108 | completed |
| ogbn-arxiv | S2_coverage_medoids | 0.0025 | 0.38330966234207153 | 0.1439939347677864 | 29 | 0.002497888900043108 | completed |
| ogbn-arxiv | S4_teacher_kd | 0.0025 | 0.3736394941806793 | 0.09214521762914955 | 17 | 0.002497888900043108 | completed |
| ogbn-arxiv | S0_current_best | 0.005 | 0.44456103444099426 | 0.25993386530317364 | 39 | 0.005001682974790809 | completed |
| ogbn-arxiv | S2_coverage_medoids | 0.005 | 0.4671933948993683 | 0.2052860957570374 | 34 | 0.005001682974790809 | completed |
| ogbn-arxiv | S4_teacher_kd | 0.005 | 0.2010781168937683 | 0.059561424615094435 | 16 | 0.005001682974790809 | completed |
| ogbn-products | S0_current_best | 0.0005 | 0.5135102868080139 | 0.18708413856183279 | 36 | 0.0005001982418338044 | completed |
| ogbn-products | S2_coverage_medoids | 0.0005 |  |  |  |  | timeout_dropped |
| ogbn-products | S4_teacher_kd | 0.0005 |  |  |  |  | timeout_dropped |
| ogbn-products | S0_current_best | 0.0025 |  |  |  |  | timeout_dropped |
| ogbn-products | S2_coverage_medoids | 0.0025 |  |  |  |  | timeout_dropped |
| ogbn-products | S4_teacher_kd | 0.0025 |  |  |  |  | timeout_dropped |
| ogbn-products | S0_current_best | 0.005 |  |  |  |  | timeout_dropped |
| ogbn-products | S2_coverage_medoids | 0.005 |  |  |  |  | timeout_dropped |
| ogbn-products | S4_teacher_kd | 0.005 |  |  |  |  | timeout_dropped |

## Diagnostics
| dataset | variant | requested_ratio | accuracy | macro_f1 | status | reason |
| --- | --- | --- | --- | --- | --- | --- |
| acm | Medoid-vs-Mean | 0.096 | 0.8682719469070435 | 0.8684150377909342 | completed |  |
| acm | CoverageMedoid | 0.096 | 0.7455146312713623 | 0.7218442956606547 | completed |  |
| imdb | PathLAD-off | 0.048 | 0.30387258529663086 | 0.23062760829925538 | completed |  |
| imdb | PathLAD-on | 0.048 | 0.28107431530952454 | 0.23565983027219772 | completed |  |
| imdb | KD-off | 0.048 | 0.28107431530952454 | 0.23565983027219772 | completed |  |
| imdb | KD-on | 0.048 | 0.25515303015708923 | 0.20060989558696746 | completed |  |

## Files

- Small CSV: `experiments/tables/sota_small_seed42.csv`
- Medium CSV: `experiments/tables/sota_medium_seed42.csv`
- Diagnostics CSV: `experiments/tables/sota_diagnostics_seed42.csv`
