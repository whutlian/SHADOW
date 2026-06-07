# Method Notes

Shadow-HGC-R-1 is a target-type node classification condensation method. It reformulates heterogeneous graph condensation as target-side directed typed relation-demand condensation, rather than source-node selection or dense synthetic adjacency learning.

The main pipeline implemented here contains:

1. leakage-safe typed feature preparation;
2. destination-row alpha normalization;
3. degree-calibrated target model input features;
4. class-wise supervised target prototypes;
5. target-target residual skeleton without top-k renormalization;
6. signed virtual residual shadows with non-negative edge weights;
7. schema-preserving materialization;
8. explicit weighted relation-linear message passing.

The one-layer relation-linear guarantee is the only theoretical scope. Deeper or attention-based backbones are empirical transfer experiments only.
