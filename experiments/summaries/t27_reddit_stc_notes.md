# T27 Reddit STC Notes

- Required Reddit STC rows are declared at 0.50% and 1.00% full-node ratios.
- Seed 42 smoke is local; seeds 1..5 are emitted by the server command and must not be inferred from seed 42.
- No smoke row is promoted.

| requested_full_node_ratio | seed | method | status | stc_objective | promotion_status | failure_reason |
|---|---|---|---|---|---|---|
| 0.005 | 42 | reddit_random_frozen_init | completed_smoke | frozen_init | not_promoted | local_smoke_not_full_reddit_run |
| 0.005 | 42 | reddit_random_trainable_delta_rho005 | completed_smoke | trainable_delta | not_promoted | local_smoke_not_full_reddit_run |
| 0.005 | 42 | reddit_random_trainable_delta_rho010 | completed_smoke | trainable_delta | not_promoted | local_smoke_not_full_reddit_run |
| 0.005 | 42 | reddit_random_gm | completed_smoke | gradient_matching | not_promoted | local_smoke_not_full_reddit_run |
| 0.005 | 42 | reddit_random_outer | completed_smoke | outer_loop | not_promoted | local_smoke_not_full_reddit_run |
| 0.005 | 42 | reddit_random_gm_plus_moment | completed_smoke | gradient_matching | not_promoted | local_smoke_not_full_reddit_run |
| 0.005 | 42 | reddit_kcenter_trainable_delta | completed_smoke | trainable_delta | not_promoted | local_smoke_not_full_reddit_run |
| 0.005 | 42 | reddit_medoid_trainable_delta | completed_smoke | trainable_delta | not_promoted | local_smoke_not_full_reddit_run |
| 0.01 | 42 | reddit_random_frozen_init | completed_smoke | frozen_init | not_promoted | local_smoke_not_full_reddit_run |
| 0.01 | 42 | reddit_random_trainable_delta_rho005 | completed_smoke | trainable_delta | not_promoted | local_smoke_not_full_reddit_run |
| 0.01 | 42 | reddit_random_trainable_delta_rho010 | completed_smoke | trainable_delta | not_promoted | local_smoke_not_full_reddit_run |
| 0.01 | 42 | reddit_random_gm | completed_smoke | gradient_matching | not_promoted | local_smoke_not_full_reddit_run |
| 0.01 | 42 | reddit_random_outer | completed_smoke | outer_loop | not_promoted | local_smoke_not_full_reddit_run |
| 0.01 | 42 | reddit_random_gm_plus_moment | completed_smoke | gradient_matching | not_promoted | local_smoke_not_full_reddit_run |
| 0.01 | 42 | reddit_kcenter_trainable_delta | completed_smoke | trainable_delta | not_promoted | local_smoke_not_full_reddit_run |
| 0.01 | 42 | reddit_medoid_trainable_delta | completed_smoke | trainable_delta | not_promoted | local_smoke_not_full_reddit_run |

- CSV: `experiments\tables\t27_stc_reddit_seed42.csv`
- Full server command: `python scripts/run_t27_stc_reddit.py --device cuda --ratios 0.005 0.01 --init current_sft_signature_random --methods frozen_init trainable_delta gradient_matching outer_loop --delta-rhos 0.05 0.10 --seeds 1 2 3 4 5`
