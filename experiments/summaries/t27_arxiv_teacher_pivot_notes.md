# T27 Arxiv Teacher Pivot Notes

- Arxiv STC condensation is blocked until a fullgraph/table teacher reaches A1 accuracy >= 0.715.
- Time-aware rows are declared as teacher-pivot work; smoke rows are not promoted.
- Correct-and-smooth residual branch is no-logits by contract; GNN teacher is upper-bound diagnostic by default.

| method | status | accuracy | macro_f1 | valid_acc | A1_passed | teacher_gate_status | promotion_status | failure_reason |
|---|---|---|---|---|---|---|---|---|
| arxiv_timeaware_sft_v5_h512 | completed_long_reference | 0.706828796576343 | 0.5045803241909133 | 0.7182120205376019 | False | blocked_below_A1 | not_promoted | arxiv_teacher_below_0.715 |
| arxiv_timeaware_sft_v5_h768 | blocked_by_teacher_gate |  |  | 0.7182120205376019 | True | A1_passed | not_promoted | arxiv_teacher_below_0.715 |
| arxiv_timeaware_sft_v5_decay_gamma005 | blocked_by_teacher_gate |  |  | 0.7182120205376019 | True | A1_passed | not_promoted | arxiv_teacher_below_0.715 |
| arxiv_timeaware_sft_v5_decay_gamma010 | blocked_by_teacher_gate |  |  | 0.7182120205376019 | True | A1_passed | not_promoted | arxiv_teacher_below_0.715 |
| arxiv_correct_smooth_no_logits | blocked_by_teacher_gate |  |  | 0.7182120205376019 | True | A1_passed | not_promoted | arxiv_teacher_below_0.715 |
| arxiv_gnn_teacher_upper_bound | blocked_by_teacher_gate |  |  | 0.7182120205376019 | True | A1_passed | upper_bound_diagnostic | upper_bound_diagnostic_not_promoted |
| arxiv_t26_best_teacher_reference | completed_long_reference | 0.706828796576343 | 0.5045803241909133 | 0.7182120205376019 | False | blocked_below_A1 | not_promoted | arxiv_teacher_below_0.715 |

- CSV: `experiments\tables\t27_arxiv_teacher_pivot_seed42.csv`
- Full server command: `python scripts/run_t27_arxiv_teacher_pivot.py --device cuda --variants year_features temporal_decay temporal_decay_year residual_no_logits --hidden-dims 512 768 --temporal-decay-gammas 0.05 0.10 --run-long --seed 42`
