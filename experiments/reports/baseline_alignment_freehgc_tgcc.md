# Baseline Alignment: FreeHGC / TGCC

This sprint did not implement or re-run FreeHGC/TGCC. The table aligns mechanism-level gaps against our current base and R+ results without inventing external numbers.

| Dataset | Baseline name | Reported metric | Reported ratio or budget | Uses meta-path | Uses teacher/training | Uses spectral/diffusion | Our best base | Our best R+ | Gap | Notes |
|---|---|---|---|---|---|---|---:|---:|---|---|
| acm | FreeHGC | paper-reported accuracy/F1; not re-run here | paper-specific condensation ratio | yes | yes | no | 0.6596 | 0.8432 | meta-path signal now partially addressed | R+ meta-path target features preserve schema and improve the IMDB failure case. |
| dblp | FreeHGC | paper-reported accuracy/F1; not re-run here | paper-specific condensation ratio | yes | yes | no | 0.8313 | 0.8370 | meta-path signal now partially addressed | R+ meta-path target features preserve schema and improve the IMDB failure case. |
| imdb | FreeHGC | paper-reported accuracy/F1; not re-run here | paper-specific condensation ratio | yes | yes | no | 0.3507 | 0.3810 | meta-path signal now partially addressed | R+ meta-path target features preserve schema and improve the IMDB failure case. |
| ogbn-arxiv | TGCC-style spectral/diffusion baseline | paper-reported accuracy; not re-run here | paper-specific budget | no | varies | yes | 0.4664 | 0.5369 | diffusion helps | Diffusion narrows arxiv gap; products needs out-of-core diffusion before fair R+ comparison. |
| ogbn-products | TGCC-style spectral/diffusion baseline | paper-reported accuracy; not re-run here | paper-specific budget | no | varies | yes | 0.5891 | 0.5891 | R+ diffusion OOM; chunking required | Diffusion narrows arxiv gap; products needs out-of-core diffusion before fair R+ comparison. |
