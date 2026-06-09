# SOTA Clean Small Seed 42

Clean rows disable KD, coverage medoids, source anchors, and diffusion. S1 rows use actual `SeHGNNLite` feature-block training.

| Dataset | Variant | Ratio | Acc | Macro-F1 | Pred classes | Condensed node ratio | Status |
|---|---|---:|---:|---:|---:|---:|---|
| acm | S1_clean_metapath_sehgnn | 0.048 | 0.8432483673095703 | 0.8443612655003866 | 3 | 0.004021202705172729 | completed |
| acm | S1_clean_metapath_sehgnn | 0.096 | 0.8800755143165588 | 0.8795819679896036 | 3 | 0.007768232498629135 | completed |
| acm | S1_clean_metapath_sehgnn | 0.12 | 0.8937677145004272 | 0.8940866192181905 | 3 | 0.0097788338512155 | completed |
| acm | S1_clean_metapath_sehgnn | 0.15 | 0.8706326484680176 | 0.869088351726532 | 3 | 0.011698044233229756 | completed |
| dblp | S0_current_best | 0.005 | 0.7838028073310852 | 0.7758937478065491 | 4 | 0.001224739742804654 | completed |
| dblp | S1_clean_APA_sehgnn | 0.005 | 0.7626760601997375 | 0.7586506754159927 | 4 | 0.000612369871402327 | completed |
| dblp | S0_current_best | 0.065 | 0.7785211205482483 | 0.7709946632385254 | 4 | 0.006047152480097979 | completed |
| dblp | S1_clean_APA_sehgnn | 0.065 | 0.7700704336166382 | 0.7602305710315704 | 4 | 0.0030235762400489894 | completed |
| dblp | S0_current_best | 0.096 | 0.7845070362091064 | 0.7770234197378159 | 4 | 0.008955909369259033 | completed |
| dblp | S1_clean_APA_sehgnn | 0.096 | 0.7644366025924683 | 0.7589197754859924 | 4 | 0.004477954684629516 | completed |
| imdb | Rpp_shadow_fusion_class_balanced_reference | 0.005 | 0.3425983786582947 | 0.3311773508787155 | 5 | 0.005929038281979458 | completed |
| imdb | S1_clean_MAM_MDM_MKM | 0.005 | 0.37164270877838135 | 0.34688123464584353 | 5 | 0.0009337068160597573 | completed |
| imdb | PathLAD_v2_only | 0.005 | 0.35352903604507446 | 0.34126800000667573 | 5 | 0.0009337068160597573 | completed |
| imdb | PathLAD_v2_plus_shadow_fusion | 0.005 | 0.3279200494289398 | 0.31737546622753143 | 5 | 0.0009337068160597573 | completed |
| imdb | Rpp_shadow_fusion_class_balanced_reference | 0.025 | 0.34134915471076965 | 0.3266520440578461 | 5 | 0.011064425770308124 | completed |
| imdb | S1_clean_MAM_MDM_MKM | 0.025 | 0.3863210380077362 | 0.3676772892475128 | 5 | 0.0015873015873015873 | completed |
| imdb | PathLAD_v2_only | 0.025 | 0.35227981209754944 | 0.3251879423856735 | 5 | 0.0015873015873015873 | completed |
| imdb | PathLAD_v2_plus_shadow_fusion | 0.025 | 0.3113678991794586 | 0.3068678677082062 | 5 | 0.0015873015873015873 | completed |
| imdb | Rpp_shadow_fusion_class_balanced_reference | 0.05 | 0.3504059910774231 | 0.33577802777290344 | 5 | 0.022549019607843137 | completed |
| imdb | S1_clean_MAM_MDM_MKM | 0.05 | 0.42410993576049805 | 0.35393159091472626 | 5 | 0.0032212885154061623 | completed |
| imdb | PathLAD_v2_only | 0.05 | 0.367582768201828 | 0.3188542157411575 | 5 | 0.0032212885154061623 | completed |
| imdb | PathLAD_v2_plus_shadow_fusion | 0.05 | 0.3372891843318939 | 0.30090869665145875 | 5 | 0.0032212885154061623 | completed |

## Notes

- DBLP clean SeHGNN is author-targeted and uses APA when available.
- IMDB Path-LAD v2 rows use train labels only with leave-one-out, row normalization, hub clipping diagnostics, and no exposed meta-path edge types.
- `PathLAD_v2_plus_shadow_fusion` is not promoted unless later wired into the graph shadow-fusion model; this script records it as a feature-block fusion diagnostic.

- CSV: `experiments\tables\sota_clean_small_seed42.csv`
