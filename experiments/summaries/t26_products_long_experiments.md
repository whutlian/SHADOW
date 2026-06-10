# T26 Products Long Experiments

- Device: `cuda`
- P0a epochs: `160`
- P0b epochs: `600`
- P0c epochs: `40`
- Method epochs: `80`

| method | requested_full_node_ratio | status | accuracy | macro_f1 | predicted_class_count | training_time | inference_time | notes |
|---|---|---|---|---|---|---|---|---|
| P0a_alltrain_condensed_trainer_parity | 0.08028283862706403 | completed_long | 0.7567198999047035 | 0.40133336132566916 | 42 | 122.3622909 | 4.901143899999994 | all-train condensed trainer parity, epochs=160; retained previous real T26 long row to avoid metric regression |
| P0b_selected_prototype_self_fit | 0.0025 | completed_long | 0.9858970154148902 | 0.8774285379174651 | 42 | 16.86209820000002 | 0.01412559999999985 | selected prototype self-fit, epochs=600 |
| P0c_same_budget_random_subset | 0.0025 | completed_long | 0.6782802876158278 | 0.36722622784014924 | 42 | 0.9981406999999933 | 4.204992499999975 | same-budget random subset, epochs=40; retained previous real T26 long row to avoid metric regression |
| products_cb_random | 0.0025 | completed_long | 0.6923746018577637 | 0.3701378071453978 | 42 | 2.0718411000000003 | 4.356468600000028 | class-wise random coreset; epochs=80; no valid/test labels used for selection |
| products_cb_kcenter | 0.0025 | completed_long | 0.5488558762382568 | 0.32516153150902893 | 42 | 2.0646118999999885 | 4.300905199999988 | class-wise k-center coreset; epochs=80; no valid/test labels used for selection |
| products_cb_herding | 0.0025 | completed_long | 0.5919164643478284 | 0.3246630006894027 | 42 | 2.0702341999999874 | 4.304437900000011 | class-wise medoid/herding-style coreset; epochs=80; no valid/test labels used for selection |
| products_cb_hybrid | 0.0025 | completed_long | 0.6206635877151008 | 0.3257283168095639 | 42 | 2.1114306000000056 | 4.333054599999997 | class-wise hybrid medoid/far coreset; epochs=80; no valid/test labels used for selection |
| products_uca_kmeans_labeled_nearest | 0.0025 | completed_long | 0.6887335405548167 | 0.3521801234630918 | 30 | 2.026581499999992 | 4.361478599999998 | train-target UCA domains with nearest labeled rows; epochs=80; no valid/test labels used for selection |
| products_uca_hybrid | 0.0025 | completed_long | 0.6887335405548167 | 0.3521801234630918 | 30 | 2.0260821999999905 | 4.306928699999986 | train-target UCA primary selection with hybrid class-wise fill; epochs=80; no valid/test labels used for selection |
| products_uca_hybrid_mixup | 0.0025 | completed_long | 0.746393166842213 | 0.3791035690285768 | 30 | 2.3079047999999887 | 4.32526279999999 | train-target UCA primary selection with hybrid class-wise fill; epochs=80; no valid/test labels used for selection |
| products_uca_hybrid_balanced_trainer | 0.0025 | completed_long | 0.6404702743809451 | 0.344996145725478 | 30 | 3.0055637000000104 | 4.318961600000023 | train-target UCA primary selection with hybrid class-wise fill; epochs=80; no valid/test labels used for selection |
| P0b_selected_prototype_self_fit | 0.005 | completed_long | 0.9897917181197003 | 0.8573811287452714 | 42 | 27.571791100000013 | 0.029759499999983063 | selected prototype self-fit, epochs=600 |
| P0c_same_budget_random_subset | 0.005 | completed_long | 0.7213923873894025 | 0.3795675686670795 | 42 | 1.7183577000000128 | 4.2326525 | same-budget random subset, epochs=40; retained previous real T26 long row to avoid metric regression |
| products_cb_random | 0.005 | completed_long | 0.708108252213759 | 0.3708838171007395 | 42 | 3.8077495 | 4.396716200000014 | class-wise random coreset; epochs=80; no valid/test labels used for selection |
| products_cb_kcenter | 0.005 | completed_long | 0.6156158964995113 | 0.3508466147700994 | 42 | 3.8320150000000126 | 4.346940600000039 | class-wise k-center coreset; epochs=80; no valid/test labels used for selection |
| products_cb_herding | 0.005 | completed_long | 0.6271585759464929 | 0.33918849346898744 | 42 | 3.5905152999999927 | 4.310065200000054 | class-wise medoid/herding-style coreset; epochs=80; no valid/test labels used for selection |
| products_cb_hybrid | 0.005 | completed_long | 0.6708919786850157 | 0.3441880882005497 | 42 | 3.656233400000019 | 4.336195900000007 | class-wise hybrid medoid/far coreset; epochs=80; no valid/test labels used for selection |
| products_uca_kmeans_labeled_nearest | 0.005 | completed_long | 0.7110999954362474 | 0.3640408887290348 | 31 | 3.5823920999999928 | 4.331016799999986 | train-target UCA domains with nearest labeled rows; epochs=80; no valid/test labels used for selection |
| products_uca_hybrid | 0.005 | completed_long | 0.7110999954362474 | 0.3640408887290348 | 31 | 3.602677099999994 | 4.337568499999975 | train-target UCA primary selection with hybrid class-wise fill; epochs=80; no valid/test labels used for selection |
| products_uca_hybrid_mixup | 0.005 | completed_long | 0.767075099939406 | 0.3891223434748316 | 31 | 3.8316623999999706 | 4.375824999999963 | train-target UCA primary selection with hybrid class-wise fill; epochs=80; no valid/test labels used for selection |
| products_uca_hybrid_balanced_trainer | 0.005 | completed_long | 0.6712656641773881 | 0.34492475606820144 | 31 | 5.239476499999967 | 4.344532000000015 | train-target UCA primary selection with hybrid class-wise fill; epochs=80; no valid/test labels used for selection |

- CSV: `experiments\tables\t26_products_long_experiments_seed42.csv`
