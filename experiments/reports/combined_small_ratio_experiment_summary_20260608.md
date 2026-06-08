# Combined Small Ratio Experiment Summary - 2026-06-08

## Scope

- Experiment A: DBLP / ACM / IMDB at target compression ratios 1.2%, 2.4%, 4.8%, 9.6%, averaged over seeds 0/1/2.
- Experiment B: DBLP / IMDB single-seed sweep from 0.5% to 12.0% with 0.5 percentage-point spacing, seed 0 only.
- Common setting: local conda environment `pytorch`, Shadow-HGC-R-1 relation-linear backbone, clipped prototype loss, raw feature mode, 500 training epochs.
- Note: the reported ratio is the requested target prototype ratio. Condensed node ratio can be higher because relation shadow nodes are added.

## Experiment A - Fixed Ratios, 3 Seeds

| Dataset | Ratio | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std | Condensed nodes seed0 |
|---|---:|---:|---:|---:|---:|---:|
| DBLP | 1.2% | 0.8264 | 0.0020 | 0.8185 | 0.0020 | 32 |
| DBLP | 2.4% | 0.8276 | 0.0007 | 0.8202 | 0.0010 | 58 |
| DBLP | 4.8% | 0.8269 | 0.0004 | 0.8193 | 0.0005 | 116 |
| DBLP | 9.6% | 0.8268 | 0.0016 | 0.8188 | 0.0017 | 234 |
| ACM | 1.2% | 0.6014 | 0.1374 | 0.5680 | 0.1565 | 52 |
| ACM | 2.4% | 0.6829 | 0.0536 | 0.6452 | 0.0678 | 62 |
| ACM | 4.8% | 0.7137 | 0.0369 | 0.7011 | 0.0440 | 111 |
| ACM | 9.6% | 0.8573 | 0.0089 | 0.8577 | 0.0093 | 218 |
| IMDB | 1.2% | 0.3376 | 0.0059 | 0.2998 | 0.0063 | 44 |
| IMDB | 2.4% | 0.3181 | 0.0064 | 0.2921 | 0.0068 | 66 |
| IMDB | 4.8% | 0.3142 | 0.0081 | 0.2997 | 0.0101 | 132 |
| IMDB | 9.6% | 0.2967 | 0.0084 | 0.2757 | 0.0046 | 263 |

### Experiment A Takeaways

- ACM behaves as expected: accuracy rises with more budget, from 0.6014 at 1.2% to 0.8573 at 9.6%.
- DBLP is almost saturated at low budget: all four ratios stay around 0.826-0.828 accuracy.
- IMDB does not improve with more budget in this setup: accuracy drops from 0.3376 at 1.2% to 0.2967 at 9.6%.

## Experiment B - DBLP / IMDB Fine Sweep, Seed 0

| Dataset | Best ratio by accuracy | Best accuracy | Macro-F1 at best accuracy | Condensed nodes | Best ratio by macro-F1 | Best macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| DBLP | 6.5% | 0.8331 | 0.8254 | 158 | 6.5% | 0.8254 |
| IMDB | 0.5% | 0.3376 | 0.2932 | 44 | 2.5% | 0.3074 |

### Accuracy Curve

- DBLP: 0.5%=0.8289; 1.0%=0.8289; 1.5%=0.8289; 2.0%=0.8306; 2.5%=0.8257; 3.0%=0.8306; 3.5%=0.8299; 4.0%=0.8275; 4.5%=0.8278; 5.0%=0.8278; 5.5%=0.8278; 6.0%=0.8292; 6.5%=0.8331; 7.0%=0.8303; 7.5%=0.8254; 8.0%=0.8285; 8.5%=0.8282; 9.0%=0.8243; 9.5%=0.8282; 10.0%=0.8310; 10.5%=0.8250; 11.0%=0.8250; 11.5%=0.8282; 12.0%=0.8282
- IMDB: 0.5%=0.3376; 1.0%=0.3376; 1.5%=0.3260; 2.0%=0.3017; 2.5%=0.3264; 3.0%=0.3226; 3.5%=0.3148; 4.0%=0.3323; 4.5%=0.3276; 5.0%=0.3339; 5.5%=0.2967; 6.0%=0.2842; 6.5%=0.2886; 7.0%=0.3011; 7.5%=0.3132; 8.0%=0.2845; 8.5%=0.2961; 9.0%=0.3151; 9.5%=0.3182; 10.0%=0.2692; 10.5%=0.2992; 11.0%=0.2914; 11.5%=0.2502; 12.0%=0.2767

### Experiment B Takeaways

- DBLP remains flat over the wider range, with a small peak at 6.5% accuracy 0.8331. This supports the saturation interpretation from Experiment A.
- IMDB is unstable and mostly worsens as budget increases. The best accuracy is tied at 0.5% and 1.0%, while the best macro-F1 appears at 2.5%.
- The contrasting behavior suggests the current relation-linear condensed graph is sufficient for ACM and DBLP under these settings, but not for IMDB; IMDB likely needs backbone/loss/feature diagnostics rather than simply more target prototypes.

## Files

- Fixed-ratio CSV: `D:\Shadow-HGC\experiments\tables\small_requested_ratio_accuracy_20260608.csv`
- Fine-sweep CSV: `D:\Shadow-HGC\experiments\tables\dblp_imdb_ratio_sweep_seed0_20260608.csv`
- This summary: `D:\Shadow-HGC\experiments\reports\combined_small_ratio_experiment_summary_20260608.md`
