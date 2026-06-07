# Experiment Protocol

## Stage 0-1

Run invariant tests first:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests -q
```

Run the toy end-to-end path:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/run_toy.py --output experiments/logs/toy/summary.json
```

The toy JSON includes directed relations, graph size, training/inference timing, schema preservation, non-negative edge-weight checks, and required relation diagnostics.

## Stage 2-3

`scripts/run_small.py` runs the local processed ACM/DBLP/IMDB PyG files over multiple seeds and writes `small_main.csv` plus executed ablations in `small_ablation.csv`. The current table includes Shadow-HGC-R-1 and simple target-feature coreset baselines: Random-HG, Herding-HG, and K-Center-HG.

`scripts/run_medium.py` runs ogbn-arxiv and ogbn-products as homogeneous special cases with explicit directed forward/reverse relations. It writes `medium_main.csv`, `medium_ablation.csv`, and the medium skeleton coverage figure. If an OGB dataset is missing or a run exceeds local resources, the script writes explicit status rows rather than omitting the failure.

## Stage 4

Always run dry-run estimates before ultra-scale execution:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/dry_run_ultra.py --output experiments/logs/scaling_stress/dry_run.json --stress
```

The large-mode cache path is train-target-only by default and refuses all-node demand caching unless debug mode is explicitly enabled. Each relation uses at most two full edge scans: degree/active-source collection, then message aggregation plus optional compact train-target edge-slice caching.
