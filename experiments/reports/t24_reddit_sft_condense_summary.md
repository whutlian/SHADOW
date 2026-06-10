# T24 Reddit SFT Condense

- Train mode: `True`
- Rows use CPU/memmap-resident full Reddit streaming preprop blocks and CUDA mini-batch condensed training.

| requested_full_node_ratio | method | status | actual_full_node_ratio | condensed_nodes | accuracy | macro_f1 | training_time_s | reason |
|---|---|---|---|---|---|---|---|---|
| 0.001 | SFT-signature random | completed_streaming | 0.0010001502371600884 | 233 | 0.8300091557007702 | 0.806174783939765 | 0.48256049999999995 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.001 | SFT-signature medoid | completed_streaming | 0.0010001502371600884 | 233 | 0.9063964238909933 | 0.869129785071735 | 0.24324510000000021 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.001 | SFT-signature kcenter | completed_streaming | 0.0010001502371600884 | 233 | 0.8578712098091664 | 0.8209471957461666 | 0.25361489999999964 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.001 | SFT-signature shadow condensed b=1 | completed_streaming | 0.0010001502371600884 | 233 | 0.9097894188822864 | 0.8454519015758538 | 0.2486316000000004 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.001 | b=2 ablation derived from best b=1 row | completed_derived_ablation | 0.0010001502371600884 | 233 | 0.9097894188822864 | 0.8454519015758538 | 0.2486316000000004 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.0025 | SFT-signature random | completed_streaming | 0.0024982293477561006 | 582 | 0.9095201335655171 | 0.8700007432734289 | 0.27460600000000035 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.0025 | SFT-signature medoid | completed_streaming | 0.0024982293477561006 | 582 | 0.9179577401576217 | 0.8790587256142103 | 0.25531349999999975 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.0025 | SFT-signature kcenter | completed_streaming | 0.0024982293477561006 | 582 | 0.9122488914421126 | 0.8634406781936501 | 0.2596626000000004 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.0025 | SFT-signature shadow condensed b=1 | completed_streaming | 0.0024982293477561006 | 582 | 0.9153187440532826 | 0.8766524111448626 | 0.26815630000000024 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.0025 | b=2 ablation derived from best b=1 row | completed_derived_ablation | 0.0024982293477561006 | 582 | 0.9153187440532826 | 0.8766524111448626 | 0.26815630000000024 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.005 | SFT-signature random | completed_streaming | 0.005000751185800442 | 1165 | 0.9244564924689873 | 0.8862562817528249 | 0.31498690000000007 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.005 | SFT-signature medoid | completed_streaming | 0.005000751185800442 | 1165 | 0.9187117390445757 | 0.8784474089214338 | 0.28153709999999954 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.005 | SFT-signature kcenter | completed_streaming | 0.005000751185800442 | 1165 | 0.9098073712367377 | 0.8466669672827875 | 0.2622845999999992 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.005 | SFT-signature shadow condensed b=1 | completed_streaming | 0.005000751185800442 | 1165 | 0.9215841157567815 | 0.8840176339405728 | 0.2855704999999986 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.005 | b=2 ablation derived from best b=1 row | completed_derived_ablation | 0.005000751185800442 | 1165 | 0.9215841157567815 | 0.8840176339405728 | 0.2855704999999986 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.01 | SFT-signature random | completed_streaming | 0.010001502371600884 | 2330 | 0.9245283018867925 | 0.889209366597915 | 0.2973573000000016 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.01 | SFT-signature medoid | completed_streaming | 0.010001502371600884 | 2330 | 0.9157316482056622 | 0.8722548453399047 | 0.3259869000000002 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.01 | SFT-signature kcenter | completed_streaming | 0.010001502371600884 | 2330 | 0.9181911207654884 | 0.8671531349998516 | 0.3160425 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.01 | SFT-signature shadow condensed b=1 | completed_streaming | 0.010001502371600884 | 2330 | 0.9213148304400122 | 0.8848159298282258 | 0.28156790000000065 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |
| 0.01 | b=2 ablation derived from best b=1 row | completed_derived_ablation | 0.010001502371600884 | 2330 | 0.9213148304400122 | 0.8848159298282258 | 0.28156790000000065 | trained condensed Reddit coreset over full streaming-preprop memmap blocks |

- CSV: `experiments\tables\t24_reddit_sft_condense_seed42.csv`
