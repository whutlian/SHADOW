# Small Results Notes

The current Stage 2 command runs Shadow-HGC-R-1 on the local processed ACM/DBLP/IMDB PyG files over three seeds and two `M_tau` settings, then writes `experiments/tables/small_main.csv`.

Current Stage 2 evidence:

1. `small_main.csv` includes Shadow-HGC-R-1, Random-HG, Herding-HG, and K-Center-HG rows with mean and standard deviation.
2. `small_ablation.csv` contains executed ablations for degree features, residual shadows, real source centroids, loss variants, relation norm calibration, and target-target `k_s` where applicable.
3. `experiments/figures/skeleton_coverage_vs_accuracy.*` is generated from the `k_s` ablation rows.

External HGCond/FreeHGC/HGC-Herd baselines remain future integration work outside the main Stage 0-4 stabilization path.
