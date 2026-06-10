# T23 Shadow-HGC SFT Method Note

T23 is an opt-in SFT enhancement stage. It does not replace the default Shadow-HGC-R-1 condensation path.

## What Changed

- Added a T23 filter-bank v3 wrapper with explicit arxiv citation-direction block names, residual feature blocks, train-label-only LabelReuse v2 names, fp16 memmap compatibility, and train-row block statistics.
- Added SAGN/GAMLP lite v3 entry names as aliases over the existing no-final-ReLU SFT heads, with label-dropout and attention-head diagnostics.
- Added a shared T23 selection score: `valid_acc + 0.05 * valid_macro_f1`.
- Added reusable SFT signature and condensation utilities for centroid, medoid, herding, and b=2 nonnegative assignment diagnostics.
- Added stage scripts and reports for arxiv, products, DBLP, ACM, and ultra dry-run accounting.

## Safety Constraints

- No logits are used as SFT input blocks.
- No KD path is promoted.
- No dense P2, bounded-edge performance row, or E-by-d edge-feature materialization is added.
- LabelReuse uses train labels only.
- Ultra-scale dry-run marks all-target caching as forbidden for papers100M/MAG and keeps train-target-only as the allowed route.

## Experiment Source Policy

The default T23 stage runner replays existing local full-edge memmap experiments where rerunning the large OGB jobs would be expensive. Replay rows keep `source_experiment` and `status` fields explicit. The code paths needed for v3 heads, label blocks, signatures, condensation, and dry-run accounting are implemented and covered by tests.
