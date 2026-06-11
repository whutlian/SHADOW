from __future__ import annotations

from shadow_hgc.reddit.operator_students import OperatorSFTTableHead, WeightedOperatorStudent


class QOCWeightedSGC(WeightedOperatorStudent):
    uses_library_normalization = False


class QOCWeightedGCN(WeightedOperatorStudent):
    uses_library_normalization = False


class QOCWeightedGraphSAGE(WeightedOperatorStudent):
    uses_library_normalization = False


QOCOperatorSFTTableHead = OperatorSFTTableHead
