# Shadow-HGC-R+ Rescue Summary

## Scope

- Seed: 42 only for all R+ sprint experiments.
- R-1 defaults remain fixed: fixed shadow budget, b=1, base features, fixed-k skeleton, no relation gate.
- R+ additions are explicit: rank diagnostics, rank-adaptive capacity, adaptive top-b, meta-path/diffusion target features, relation gate, coverage skeleton, and logit-adjusted loss.

## Main Outcomes

- IMDB rescue: best base `0.3507` / macro-F1 `0.2956`; best full R+ `0.3810` / macro-F1 `0.3403`; best overall `0.3810` from `full_rplus` + `clipped` at `0.5%`.
- ogbn-arxiv: best base `0.4664`; best included setting `0.5369` from `diffusion_X0X1X2_highpass_coverage` + `sqrt_weighted_logit_adjusted` at `6.0%`.
- ogbn-products: best base `0.5891`; best included setting `0.5891` from `base` + `sqrt_weighted` at `12.0%`.
- acm regression: best `rplus` at ratio `9.6%`, accuracy `0.8432`, macro-F1 `0.8462`.
- dblp regression: best `rplus` at ratio `6.5%`, accuracy `0.8370`, macro-F1 `0.8299`.

## OOM / Resource Signals

- ogbn-products `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted` at `0.5%`: `oom`. Log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p005_seed42.json`
- ogbn-products `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted` at `2.0%`: `oom`. Log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p02_seed42.json`
- ogbn-products `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted` at `6.0%`: `oom`. Log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p06_seed42.json`
- ogbn-products `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted` at `12.0%`: `oom`. Log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_r0p12_seed42.json`
- ogbn-products `diffusion_X0X1` / `sqrt_weighted` at `6.0%`: `oom`. Log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1_sqrt_weighted_r0p06_seed42.json`
- ogbn-products `diffusion_X0X1X2` / `sqrt_weighted` at `6.0%`: `oom`. Log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_sqrt_weighted_r0p06_seed42.json`
- ogbn-products `diffusion_highpass_coverage_adaptive` / `sqrt_weighted` at `6.0%`: `oom`. Log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_highpass_coverage_adaptive_sqrt_weighted_r0p06_seed42.json`
- ogbn-products `diffusion_X0X1X2_highpass_coverage` / `sqrt_weighted_logit_adjusted` at `6.0%`: `oom`. Log: `experiments\logs\medium_diffusion_rplus_seed42\ogbn-products_diffusion_X0X1X2_highpass_coverage_sqrt_weighted_logit_adjusted_r0p06_seed42.json`

## Hypothesis Answers

- DBLP flatness: supported. The DBLP relation keeps low reconstruction error (`max 0.0361`), and regression accuracy stays strong at low/high ratios, indicating early rank saturation.
- IMDB rescue: partially successful. Base IMDB relation reconstruction errors were high, while full R+ reduced relation reconstruction errors to near-zero in the rescue grid and improved accuracy/macro-F1 without class-count collapse.
- Medium diffusion: supported for ogbn-arxiv. Diffusion + coverage/logit adjustment reached `0.5369`, above the current base `0.4664`. ogbn-products diffusion is a resource failure under in-memory features and must move to chunked/memmap diffusion before Stage 5 scaling.
- Core idea preservation: all runs kept original exposed node/edge schema, nonnegative edge weights, destination-row alpha normalization, and the custom weighted relation-linear path.

## Source Tables

- `experiments/tables/rank_diagnostics_small_medium_seed42.csv`
- `experiments/tables/imdb_rescue_rplus_seed42.csv`
- `experiments/tables/medium_diffusion_rplus_seed42.csv`
- `experiments/tables/acm_dblp_rplus_regression_seed42.csv`
