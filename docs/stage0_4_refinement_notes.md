# Stage 0-4 Refinement Notes

Date: 2026-06-07

## Verification

- Full invariant/test suite before long experiments: `57 passed`.
- After train-target-only demand and memory-efficient relation MLP changes, targeted regression tests passed for toy, pipeline options, skeleton/prototype, relation MLP chunking, and small script paths.
- Final artifacts are regenerated from JSON logs with `scripts/build_tables.py`.

## Reboot / OOM Investigation

Windows event evidence showed two separate issues:

- `2026-06-07 02:11:57`: Resource-Exhaustion-Detector reported low virtual memory; `python.exe` consumed about `113,209,556,992` bytes of virtual memory.
- `2026-06-07 13:39`: the reboot itself was a BlueScreen / bugcheck `0xD1`, with minidump `C:\WINDOWS\Minidump\060726-14140-01.dmp`. There was no same-time Resource-Exhaustion event.

The interrupted run had completed ACM and DBLP refined small runs and reached IMDB `M_tau=64`. To continue safely, later long runs were executed in separate processes with a 24GiB private-memory guard.

Root cause found during medium products profiling: the previous medium path materialized relation demand for all target nodes. For `ogbn-products` with `feature_dim=128`, this pushed pre-training RSS to about `21.7GB`; PyTorch/Windows retained freed CPU allocations, and inference pushed private memory near `29.7GB`.

Fixes:

- Condensation demand is now materialized only for train target rows.
- Prototype signatures consume train-only demand rows through `signature_idx`.
- Skeleton/residual code maps global train node ids to train-demand rows.
- Relation MLP chunked inference avoids full encoded concatenation.
- Experiment scripts can resume with `--skip-existing`.

After the fix, `ogbn-products` pre-training RSS dropped to about `16.5GB`; refined products runs completed under the guard with peaks around `22.9-25.5GB`.

## Stage 1 Toy

Artifacts:

- `experiments/logs/toy/summary_refined.json`
- `experiments/logs/toy/summary_refined_private_shadow.json`
- `experiments/logs/toy/summary_refined_self_only.json`
- `experiments/logs/toy/summary_refined_full_graph.json`

All four toy modes reached accuracy and macro-F1 of `1.0`. Main toy training loss decreased from about `0.63` to near zero, prototype train accuracy was `1.0`, schema preservation was true, and all edge weights were non-negative. Private-shadow reconstruction error was `0.0` for all toy relations.

## Stage 2 Small

Artifacts:

- `experiments/logs/small_refined/*.json`
- `experiments/tables/small_main.csv`
- `experiments/tables/small_ablation.csv`

Settings used: raw features, `M_tau in {32,64,128}`, `min_proto_per_class=4`, `degree_scale=0.1`, clipped loss, relation MLP, 500 epochs, seeds `0 1 2`.

Summary:

- ACM: Shadow-HGC-R-1 is stable around `0.754-0.758` accuracy; full-graph upper bound is `0.799`. Random/K-Center are competitive at `M_tau=128`, so ACM is not a clean win but no class collapse occurs.
- DBLP: Shadow-HGC-R-1 reaches `0.706` at `M_tau=32`, beating Random/Herding/K-Center at the same budget and close to self-only. Full-graph upper bound is `0.747`.
- IMDB: self-only and full-graph diagnostics show the current feature/backbone setting is weak. Shadow-HGC-R-1 does not beat self-only there; the likely bottleneck is feature/backbone quality rather than shadow reconstruction alone.

The ablation table includes degree/mean-only, residual on/off, real-source centroid, private-shadow upper bound, loss variants, relation-linear vs relation-MLP, and `k_s` variants where applicable.

## Stage 3 Medium

Artifacts:

- `experiments/logs/medium_refined/*.json`
- `experiments/tables/medium_main.csv`
- `experiments/tables/medium_ablation.csv`

Settings used: random projection `feature_dim=128`, `min_proto_per_class=4`, `degree_scale=0.1`, `sqrt_weighted` loss, relation MLP, 500 epochs.

Main results:

- `ogbn-arxiv`: `M_tau=200/400/800` gives accuracy `0.2499/0.2478/0.2818`, macro-F1 `0.1400/0.1478/0.1518`, and `40` predicted classes.
- `ogbn-products`: `M_tau=500/1000/2000` gives accuracy `0.3459/0.4103/0.4685`, macro-F1 `0.1119/0.1466/0.1745`, and `42` predicted classes.

Sanity diagnostics:

- `ogbn-arxiv` full-graph same-backbone ran under the guard: accuracy `0.6079`, macro-F1 `0.3771`, indicating the condensation gap remains large.
- `ogbn-products` full-graph same-backbone was marked `resource_guard_infeasible` under the 24GiB private-memory guard.
- Self-only and private-shadow sanity logs are present for both medium datasets.
- Class collapse is fixed relative to earlier smoke results: medium runs predict 40 or 42 classes instead of one class.

The `k_s` ablation is complete for both datasets. Skeleton coverage increases monotonically with `k_s`; accuracy is not monotonic, so the current evidence supports reporting a topology/size trade-off rather than tuning `k_s` as a pure accuracy knob.

## Stage 4 Dry Run

Artifacts:

- `experiments/logs/scaling_stress/dry_run_refined.json`
- `experiments/logs/scaling_stress/dry_run_refined_stress.json`

The refined dry run reports relation-specific memory, sorted-search id-index mode, edge-slice cache bytes, edge-slice dtype, disk-spill status, and total expected full edge scans. With the requested parameters, total expected full edge scans are `6`.

## Current Interpretation

The original low-accuracy smoke results were mostly caused by smoke settings: tiny feature dimensions, too-small prototype and shadow budgets, weighted-loss class imbalance, and missing diagnostics. The refined settings remove the main class-collapse failure mode and expose more precise bottlenecks:

- On ACM/DBLP, Shadow-HGC-R-1 is competitive with simple target coreset baselines at realistic budgets.
- On IMDB, feature/backbone quality is the main blocker because self-only is stronger than relation condensation.
- On ogbn-arxiv, full-graph same-backbone is much stronger than condensed runs, so target prototype/shadow compression is the main remaining gap.
- On ogbn-products, increasing `M_tau` improves accuracy and macro-F1, supporting the budget fix.

No new main method module was introduced; changes are implementation, diagnostics, budget, feature/loss, baseline, and memory-honesty refinements within Shadow-HGC-R-1.
