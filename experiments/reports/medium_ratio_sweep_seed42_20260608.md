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
| ogbn-arxiv | 12.0% | 0.4697 | 0.3103 | 16251 | 40 |
| ogbn-products | 12.0% | 0.5891 | 0.2643 | 34174 | 40 |

## Accuracy Curve

- ogbn-arxiv: 0.5%=0.4293; 1.0%=0.4051; 1.5%=0.4003; 2.0%=0.4021; 2.5%=0.3997; 3.0%=0.3868; 3.5%=0.3860; 4.0%=0.3976; 4.5%=0.3800; 5.0%=0.4003; 5.5%=0.4014; 6.0%=0.3916; 6.5%=0.4091; 7.0%=0.4124; 7.5%=0.4065; 8.0%=0.4180; 8.5%=0.4212; 9.0%=0.4290; 9.5%=0.4414; 10.0%=0.4355; 10.5%=0.4457; 11.0%=0.4451; 11.5%=0.4641; 12.0%=0.4697
- ogbn-products: 0.5%=0.4295; 1.0%=0.4253; 1.5%=0.4425; 2.0%=0.4744; 2.5%=0.4871; 3.0%=0.5007; 3.5%=0.5113; 4.0%=0.5249; 4.5%=0.5338; 5.0%=0.5389; 5.5%=0.5447; 6.0%=0.5476; 6.5%=0.5604; 7.0%=0.5576; 7.5%=0.5629; 8.0%=0.5649; 8.5%=0.5676; 9.0%=0.5753; 9.5%=0.5732; 10.0%=0.5793; 10.5%=0.5799; 11.0%=0.5858; 11.5%=0.5830; 12.0%=0.5891

## Files

- CSV: `experiments\tables\medium_ratio_sweep_seed42_20260608.csv`
- Logs: `experiments\logs\medium_ratio_sweep_seed42_20260608`
- Summary: `experiments\reports\medium_ratio_sweep_seed42_20260608.md`
