# Shadow-HGC-R-1

Shadow-HGC-R-1 condenses target-type node classification by factorizing target-side directed typed relation demands into sparse schema-preserving shadow nodes, with target-target relations decomposed into a prototype residual skeleton plus signed residual shadow features.

## Environment

Use the local Conda environment named `pytorch`. On this Windows workspace, direct invocation avoids Conda activation temp-file races:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests -q
```

## Current Runnable Commands

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/run_toy.py --output experiments/logs/toy/summary.json
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/dry_run_ultra.py --output experiments/logs/scaling_stress/dry_run.json --stress
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/run_small.py
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/run_medium.py
```

`run_toy.py` is the implemented toy end-to-end method path. `run_small.py` runs the local processed ACM/DBLP/IMDB PyG files and writes mean/std CSV artifacts plus executed ablations. `run_medium.py` runs ogbn-arxiv and ogbn-products as homogeneous special cases with directed forward/reverse relations and writes medium main/`k_s` ablation tables.

## Scope Guard

The main method is fixed as Shadow-HGC-R-1. Do not add new main modules. Optional variants such as `b=2`, PCA, meta-path sketches, institution-aware features, or HGT/HAN transfer are ablations only.
