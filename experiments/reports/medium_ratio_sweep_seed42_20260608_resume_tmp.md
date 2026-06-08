# Medium Ratio Sweep Summary

## Scope

- Datasets: ogbn-arxiv, ogbn-products.
- Seed: 42.
- Ratios: 0.5% to 12.0%, 0.5 percentage-point spacing unless overridden.
- Setting: Shadow-HGC-R-1, relation-linear, sqrt-weighted prototype loss, random projection, 500 epochs.
- Ratio is requested target prototype ratio; condensed node ratio is higher because shadow nodes are added.

## Best Points

| Dataset | Best ratio by accuracy | Accuracy | Macro-F1 | Condensed nodes | Predicted classes |
|---|---:|---:|---:|---:|---:|
| ogbn-products | 12.0% | 0.5891 | 0.2643 | 34174 | 40 |

## Accuracy Curve

- ogbn-products: 12.0%=0.5891

## Files

- CSV: `experiments\tables\medium_ratio_sweep_seed42_20260608_resume_tmp.csv`
- Logs: `experiments\logs\medium_ratio_sweep_seed42_20260608`
- Summary: `experiments\reports\medium_ratio_sweep_seed42_20260608_resume_tmp.md`
