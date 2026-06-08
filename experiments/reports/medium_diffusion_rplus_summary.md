# Medium Diffusion R+ Summary

## Scope

- Datasets: ogbn-arxiv, ogbn-products.
- Seed: 42 only.
- Ratios: 0.5%, 2.0%, 6.0%, 12.0%.
- Main comparison: base vs diffusion_X0X1X2_highpass_coverage.
- Ratio 6.0% includes diffusion depth, adaptive shadow, and logit-adjustment ablations.

## Results

| Dataset | Variant | Loss | Ratio | Acc | Macro-F1 | Pred classes | Skel cov | Recon err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ogbn-arxiv | base | sqrt_weighted | 0.5000% | 0.4343 | 0.3015 | 40 | 0.5117 | 0.4477 |
| ogbn-arxiv | base | sqrt_weighted | 2.0000% | 0.3958 | 0.2668 | 40 | 0.5035 | 0.5114 |
| ogbn-arxiv | base | sqrt_weighted | 6.0000% | 0.3921 | 0.2564 | 40 | 0.6893 | 0.5849 |
| ogbn-arxiv | base | sqrt_weighted | 12.0000% | 0.4664 | 0.3060 | 40 | 0.7904 | 0.5619 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 0.5000% | 0.5189 | 0.3614 | 40 | 0.6080 | 0.4855 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 2.0000% | 0.4961 | 0.3368 | 40 | 0.6009 | 0.5454 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 6.0000% | 0.5043 | 0.3318 | 40 | 0.7020 | 0.6360 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 12.0000% | 0.5216 | 0.3456 | 40 | 0.7662 | 0.6099 |
| ogbn-arxiv | diffusion_X0X1 | sqrt_weighted | 6.0000% | 0.4669 | 0.3062 | 40 | 0.7265 | 0.6140 |
| ogbn-arxiv | diffusion_X0X1X2 | sqrt_weighted | 6.0000% | 0.4922 | 0.3376 | 40 | 0.7311 | 0.6049 |
| ogbn-arxiv | diffusion_highpass_coverage_adaptive | sqrt_weighted | 6.0000% | 0.4924 | 0.3273 | 40 | 0.7020 | 0.7360 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_coverage | sqrt_weighted_logit_adjusted | 6.0000% | 0.5369 | 0.3402 | 39 | 0.7020 | 0.6360 |
| ogbn-products | base | sqrt_weighted | 0.5000% | 0.4335 | 0.1954 | 41 | 0.6026 | 0.4129 |
| ogbn-products | base | sqrt_weighted | 2.0000% | 0.4697 | 0.2154 | 41 | 0.5233 | 0.5604 |
| ogbn-products | base | sqrt_weighted | 6.0000% | 0.5501 | 0.2451 | 40 | 0.4780 | 0.5264 |
| ogbn-products | base | sqrt_weighted | 12.0000% | 0.5891 | 0.2643 | 40 | 0.4380 | 0.3558 |
- ogbn-arxiv best: `0.5369` from `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted_logit_adjusted` at `6.0000%`.
- ogbn-products best: `0.5891` from `base` / `sqrt_weighted` at `12.0000%`.

## OOM / OOT / Guard Failures

| Dataset | Variant | Loss | Ratio | Status | Log |
|---|---|---|---:|---|---|
| ogbn-products | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 0.5% | oom | `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p005_seed42.json` |
| ogbn-products | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 2.0% | oom | `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p02_seed42.json` |
| ogbn-products | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 6.0% | oom | `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p06_seed42.json` |
| ogbn-products | diffusion_X0X1X2_highpass_coverage | sqrt_weighted | 12.0% | oom | `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p12_seed42.json` |
| ogbn-products | diffusion_X0X1 | sqrt_weighted | 6.0% | oom | `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1_sqrt_weighted_r0p06_seed42.json` |
| ogbn-products | diffusion_X0X1X2 | sqrt_weighted | 6.0% | oom | `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_sqrt_weighted_r0p06_seed42.json` |
| ogbn-products | diffusion_highpass_coverage_adaptive | sqrt_weighted | 6.0% | oom | `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_highpass_coverage_adaptive_sqrt_weighted_r0p06_seed42.json` |
| ogbn-products | diffusion_X0X1X2_highpass_coverage | sqrt_weighted_logit_adjusted | 6.0% | oom | `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_logit_adjusted_r0p06_seed42.json` |

## Files

- CSV: `experiments\tables\medium_diffusion_rplus_seed42.csv`
- Report: `experiments\reports\medium_diffusion_rplus_summary.md`
