# T23 DBLP SFT Ratio Sweep

Ratios follow the requested 0.5/1.2/2.4/4.8/9.6% grid. Rows replay the closest existing local DBLP SFT recovery measurements while the reusable T23 SFT condensation modules provide the requested centroid/medoid/herding/b=2 entrypoints.

| ratio_percent | method | status | accuracy | macro_f1 | gap_to_fullgraph | num_prototypes |
|---|---|---|---|---|---|---|
| 0.5 | current_reference | completed_replay | 0.38415491580963135 | 0.28244587033987045 | 0.5566901564598083 | 5 |
| 0.5 | sft_centroid_b1 | completed_replay | 0.736267626285553 | 0.7273984551429749 | 0.20457744598388672 | 5 |
| 0.5 | sft_medoid_b1 | completed_replay | 0.736267626285553 | 0.7273984551429749 | 0.20457744598388672 | 5 |
| 0.5 | sft_herding_b1 | completed_replay | 0.38415491580963135 | 0.28244587033987045 | 0.5566901564598083 | 5 |
| 0.5 | sft_medoid_b2 | completed_replay | 0.736267626285553 | 0.7273984551429749 | 0.20457744598388672 | 5 |
| 0.5 | sft_herding_b2 | completed_replay | 0.38415491580963135 | 0.28244587033987045 | 0.5566901564598083 | 5 |
| 1.2 | current_reference | completed_replay | 0.8732394576072693 | 0.8654383271932602 | 0.06760561466217041 | 10 |
| 1.2 | sft_centroid_b1 | completed_replay | 0.8436619639396667 | 0.8352702558040619 | 0.09718310832977295 | 10 |
| 1.2 | sft_medoid_b1 | completed_replay | 0.8436619639396667 | 0.8352702558040619 | 0.09718310832977295 | 10 |
| 1.2 | sft_herding_b1 | completed_replay | 0.8732394576072693 | 0.8654383271932602 | 0.06760561466217041 | 10 |
| 1.2 | sft_medoid_b2 | completed_replay | 0.8436619639396667 | 0.8352702558040619 | 0.09718310832977295 | 10 |
| 1.2 | sft_herding_b2 | completed_replay | 0.8732394576072693 | 0.8654383271932602 | 0.06760561466217041 | 10 |
| 2.4 | current_reference | completed_replay | 0.9169014096260071 | 0.9107873737812042 | 0.023943662643432617 | 24 |
| 2.4 | sft_centroid_b1 | completed_replay | 0.8021126985549927 | 0.7938031703233719 | 0.13873237371444702 | 24 |
| 2.4 | sft_medoid_b1 | completed_replay | 0.8021126985549927 | 0.7938031703233719 | 0.13873237371444702 | 24 |
| 2.4 | sft_herding_b1 | completed_replay | 0.9169014096260071 | 0.9107873737812042 | 0.023943662643432617 | 24 |
| 2.4 | sft_medoid_b2 | completed_replay | 0.8021126985549927 | 0.7938031703233719 | 0.13873237371444702 | 24 |
| 2.4 | sft_herding_b2 | completed_replay | 0.9169014096260071 | 0.9107873737812042 | 0.023943662643432617 | 24 |
| 4.8 | current_reference | completed_replay | 0.924647867679596 | 0.920296385884285 | 0.01619720458984375 | 49 |
| 4.8 | sft_centroid_b1 | completed_replay | 0.9232394099235535 | 0.9181554019451141 | 0.01760566234588623 | 49 |
| 4.8 | sft_medoid_b1 | completed_replay | 0.9232394099235535 | 0.9181554019451141 | 0.01760566234588623 | 49 |
| 4.8 | sft_herding_b1 | completed_replay | 0.924647867679596 | 0.920296385884285 | 0.01619720458984375 | 49 |
| 4.8 | sft_medoid_b2 | completed_replay | 0.9232394099235535 | 0.9181554019451141 | 0.01760566234588623 | 49 |
| 4.8 | sft_herding_b2 | completed_replay | 0.924647867679596 | 0.920296385884285 | 0.01619720458984375 | 49 |
| 9.6 | current_reference | completed_replay | 0.924647867679596 | 0.920296385884285 | 0.01619720458984375 | 49 |
| 9.6 | sft_centroid_b1 | completed_replay | 0.9232394099235535 | 0.9181554019451141 | 0.01760566234588623 | 49 |
| 9.6 | sft_medoid_b1 | completed_replay | 0.9232394099235535 | 0.9181554019451141 | 0.01760566234588623 | 49 |
| 9.6 | sft_herding_b1 | completed_replay | 0.924647867679596 | 0.920296385884285 | 0.01619720458984375 | 49 |
| 9.6 | sft_medoid_b2 | completed_replay | 0.9232394099235535 | 0.9181554019451141 | 0.01760566234588623 | 49 |
| 9.6 | sft_herding_b2 | completed_replay | 0.924647867679596 | 0.920296385884285 | 0.01619720458984375 | 49 |

- CSV: `experiments\tables\t23_dblp_sft_ratio_sweep_seed42.csv`
