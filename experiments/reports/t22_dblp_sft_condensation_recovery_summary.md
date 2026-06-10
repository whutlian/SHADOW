# T2.2 DBLP SFT Condensation Recovery

| ratio | recovery_row | status | accuracy | full_to_shadow_gap | reason |
|---|---|---|---|---|---|
| 0.005 | identity_condensed_sft_replay | completed_diagnostic | 0.9408450722694397 | 0.0 | identity replay of full SFT block signature |
| 0.005 | prototype_oracle_sft_block_signature | completed_diagnostic | 0.736267626285553 | 0.20457744598388672 | prototype SFT trained with full-train fitted block stats |
| 0.005 | shadow_condensed_sft_block_signature | completed_diagnostic | 0.38415491580963135 | 0.5566901564598083 | nearest signed shadow reconstruction over SFT block signatures |
| 0.01 | identity_condensed_sft_replay | completed_diagnostic | 0.9408450722694397 | 0.0 | identity replay of full SFT block signature |
| 0.01 | prototype_oracle_sft_block_signature | completed_diagnostic | 0.8436619639396667 | 0.09718310832977295 | prototype SFT trained with full-train fitted block stats |
| 0.01 | shadow_condensed_sft_block_signature | completed_diagnostic | 0.8732394576072693 | 0.06760561466217041 | nearest signed shadow reconstruction over SFT block signatures |
| 0.025 | identity_condensed_sft_replay | completed_diagnostic | 0.9408450722694397 | 0.0 | identity replay of full SFT block signature |
| 0.025 | prototype_oracle_sft_block_signature | completed_diagnostic | 0.8021126985549927 | 0.13873237371444702 | prototype SFT trained with full-train fitted block stats |
| 0.025 | shadow_condensed_sft_block_signature | promoted | 0.9169014096260071 | 0.023943662643432617 | nearest signed shadow reconstruction over SFT block signatures |
| 0.05 | identity_condensed_sft_replay | completed_diagnostic | 0.9408450722694397 | 0.0 | identity replay of full SFT block signature |
| 0.05 | prototype_oracle_sft_block_signature | completed_diagnostic | 0.9232394099235535 | 0.01760566234588623 | prototype SFT trained with full-train fitted block stats |
| 0.05 | shadow_condensed_sft_block_signature | promoted | 0.924647867679596 | 0.01619720458984375 | nearest signed shadow reconstruction over SFT block signatures |

- CSV: `experiments\tables\t22_dblp_sft_condensation_recovery_seed42.csv`
