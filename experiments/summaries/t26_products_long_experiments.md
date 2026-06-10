# T26 Products Long Experiments

- Device: `cuda`
- P0a epochs: `160`
- P0b epochs: `600`
- P0c epochs: `40`

| method | requested_full_node_ratio | status | accuracy | macro_f1 | predicted_class_count | training_time | inference_time | notes |
|---|---|---|---|---|---|---|---|---|
| P0a_alltrain_condensed_trainer_parity | 0.08028283862706403 | completed_long | 0.7567198999047035 | 0.40133336132566916 | 42 | 122.3622909 | 4.901143899999994 | all-train condensed trainer parity, epochs=160 |
| P0b_selected_prototype_self_fit | 0.0025 | completed_long | 0.9844211216792391 | 0.8787512736099667 | 42 | 15.204109700000004 | 0.014061400000002777 | selected prototype self-fit, epochs=600 |
| P0c_same_budget_random_subset | 0.0025 | completed_long | 0.6782802876158278 | 0.36722622784014924 | 42 | 0.9981406999999933 | 4.204992499999975 | same-budget random subset, epochs=40 |
| P0b_selected_prototype_self_fit | 0.005 | completed_long | 0.9842759529101835 | 0.8445929937497246 | 42 | 25.841221599999983 | 0.027145300000000816 | selected prototype self-fit, epochs=600 |
| P0c_same_budget_random_subset | 0.005 | completed_long | 0.7213923873894025 | 0.3795675686670795 | 42 | 1.7183577000000128 | 4.2326525 | same-budget random subset, epochs=40 |

- CSV: `experiments\tables\t26_products_long_experiments_seed42.csv`
