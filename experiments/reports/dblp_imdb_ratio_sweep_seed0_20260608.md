# DBLP/IMDB Ratio Sweep Seed0 Summary

Generated: 2026-06-07T17:48:18Z

## Setup

- Datasets: DBLP, IMDB
- Seed: 0
- Ratios: 0.5% to 12.0%, step 0.5 percentage points
- Model: `relation_linear`
- Loss: `clipped`
- Features: raw
- Ratio definition: requested target prototypes / train target nodes

## Best Points

| dataset | best_ratio | accuracy | macro_f1 | requested_budget | effective_prototypes | shadow_nodes | condensed_nodes | actual_total_node_ratio_to_train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DBLP | 6.5% | 0.8331 | 0.8254 | 79 | 79 | 79 | 158 | 12.98% |
| IMDB | 0.5% | 0.3376 | 0.2932 | 20 | 20 | 24 | 44 | 3.21% |

## Curve

| dataset | ratio | requested_budget | effective_prototypes | shadow_nodes | condensed_nodes | accuracy | macro_f1 | shadow_recon_err_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DBLP | 0.5% | 16 | 16 | 16 | 32 | 0.8289 | 0.8209 | 0.0132 |
| DBLP | 1.0% | 16 | 16 | 16 | 32 | 0.8289 | 0.8209 | 0.0132 |
| DBLP | 1.5% | 18 | 18 | 18 | 36 | 0.8289 | 0.8210 | 0.0062 |
| DBLP | 2.0% | 24 | 24 | 24 | 48 | 0.8306 | 0.8230 | 0.0163 |
| DBLP | 2.5% | 30 | 30 | 30 | 60 | 0.8257 | 0.8181 | 0.0153 |
| DBLP | 3.0% | 37 | 37 | 37 | 74 | 0.8306 | 0.8228 | 0.0242 |
| DBLP | 3.5% | 43 | 43 | 43 | 86 | 0.8299 | 0.8228 | 0.0070 |
| DBLP | 4.0% | 49 | 49 | 49 | 98 | 0.8275 | 0.8205 | 0.0584 |
| DBLP | 4.5% | 55 | 55 | 55 | 110 | 0.8278 | 0.8209 | 0.1149 |
| DBLP | 5.0% | 61 | 61 | 61 | 122 | 0.8278 | 0.8200 | 0.1296 |
| DBLP | 5.5% | 67 | 67 | 67 | 134 | 0.8278 | 0.8208 | 0.1233 |
| DBLP | 6.0% | 73 | 73 | 73 | 146 | 0.8292 | 0.8212 | 0.0141 |
| DBLP | 6.5% | 79 | 79 | 79 | 158 | 0.8331 | 0.8254 | 0.0890 |
| DBLP | 7.0% | 85 | 85 | 85 | 170 | 0.8303 | 0.8224 | 0.0040 |
| DBLP | 7.5% | 91 | 91 | 91 | 182 | 0.8254 | 0.8176 | 0.0033 |
| DBLP | 8.0% | 97 | 97 | 97 | 194 | 0.8285 | 0.8205 | 0.0598 |
| DBLP | 8.5% | 103 | 103 | 103 | 206 | 0.8282 | 0.8202 | 0.0034 |
| DBLP | 9.0% | 110 | 110 | 110 | 220 | 0.8243 | 0.8162 | 0.0002 |
| DBLP | 9.5% | 116 | 116 | 116 | 232 | 0.8282 | 0.8201 | 0.0003 |
| DBLP | 10.0% | 122 | 122 | 122 | 244 | 0.8310 | 0.8241 | 0.0213 |
| DBLP | 10.5% | 128 | 128 | 128 | 256 | 0.8250 | 0.8169 | 0.0063 |
| DBLP | 11.0% | 134 | 134 | 134 | 268 | 0.8250 | 0.8172 | 0.0830 |
| DBLP | 11.5% | 140 | 140 | 140 | 280 | 0.8282 | 0.8207 | 0.0159 |
| DBLP | 12.0% | 146 | 146 | 146 | 292 | 0.8282 | 0.8210 | 0.0367 |
| IMDB | 0.5% | 20 | 20 | 24 | 44 | 0.3376 | 0.2932 | 0.9053 |
| IMDB | 1.0% | 20 | 20 | 24 | 44 | 0.3376 | 0.2932 | 0.9053 |
| IMDB | 1.5% | 21 | 21 | 24 | 45 | 0.3260 | 0.2983 | 0.9306 |
| IMDB | 2.0% | 27 | 27 | 27 | 54 | 0.3017 | 0.2747 | 0.9424 |
| IMDB | 2.5% | 34 | 34 | 36 | 70 | 0.3264 | 0.3074 | 0.9746 |
| IMDB | 3.0% | 41 | 41 | 42 | 83 | 0.3226 | 0.2792 | 0.9854 |
| IMDB | 3.5% | 48 | 48 | 48 | 96 | 0.3148 | 0.2859 | 0.9860 |
| IMDB | 4.0% | 55 | 55 | 57 | 112 | 0.3323 | 0.2932 | 0.9704 |
| IMDB | 4.5% | 62 | 62 | 63 | 125 | 0.3276 | 0.2904 | 0.9844 |
| IMDB | 5.0% | 69 | 69 | 69 | 138 | 0.3339 | 0.2963 | 0.9843 |
| IMDB | 5.5% | 75 | 75 | 75 | 150 | 0.2967 | 0.2649 | 0.9783 |
| IMDB | 6.0% | 82 | 82 | 84 | 166 | 0.2842 | 0.2708 | 0.9707 |
| IMDB | 6.5% | 89 | 89 | 90 | 179 | 0.2886 | 0.2689 | 0.9842 |
| IMDB | 7.0% | 96 | 96 | 96 | 192 | 0.3011 | 0.2863 | 0.9831 |
| IMDB | 7.5% | 103 | 103 | 105 | 208 | 0.3132 | 0.2737 | 0.9672 |
| IMDB | 8.0% | 110 | 110 | 111 | 221 | 0.2845 | 0.2702 | 0.9833 |
| IMDB | 8.5% | 117 | 117 | 117 | 234 | 0.2961 | 0.2757 | 0.9827 |
| IMDB | 9.0% | 123 | 123 | 123 | 246 | 0.3151 | 0.2914 | 0.9773 |
| IMDB | 9.5% | 130 | 129 | 129 | 258 | 0.3182 | 0.2974 | 0.9863 |
| IMDB | 10.0% | 137 | 137 | 138 | 275 | 0.2692 | 0.2593 | 0.9736 |
| IMDB | 10.5% | 144 | 144 | 144 | 288 | 0.2992 | 0.2749 | 0.9871 |
| IMDB | 11.0% | 151 | 151 | 153 | 304 | 0.2914 | 0.2655 | 0.9779 |
| IMDB | 11.5% | 158 | 158 | 159 | 317 | 0.2502 | 0.2378 | 0.9671 |
| IMDB | 12.0% | 165 | 165 | 165 | 330 | 0.2767 | 0.2689 | 0.9803 |

## Interpretation

- DBLP is flat across this range: accuracy stays around 0.824-0.833. The best seed0 point is 6.5%, but the gain over low ratios is small.
- IMDB peaks at the lowest tested ratios, 0.5% and 1.0%, then fluctuates downward. Increasing the target prototype budget does not fix the IMDB bottleneck.
- IMDB shadow reconstruction error remains high across the sweep, consistent with the earlier diagnosis that IMDB is a relation/shadow reconstruction failure case rather than a simple under-budget case.

Source CSV: `experiments/tables/dblp_imdb_ratio_sweep_seed0_20260608.csv`