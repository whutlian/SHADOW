from __future__ import annotations

from shadow_hgc.data.small import load_processed_small_dataset_full_schema
from shadow_hgc.fullgraph.metapath_specs import available_metapath_specs


def test_dblp_full_schema_metapaths_are_detected_and_apcpa_is_reported():
    graph = load_processed_small_dataset_full_schema("dblp")

    available, skipped = available_metapath_specs("dblp", graph.relations, graph.target_type)

    assert {"APA", "APVPA", "APTPA"}.issubset(available)
    assert "APCPA" in skipped
    assert skipped["APCPA"] == "schema_missing"
