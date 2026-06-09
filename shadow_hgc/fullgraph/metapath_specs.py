from __future__ import annotations

from shadow_hgc.data.schemas import DirectedRelation


METAPATH_NAMES: dict[str, list[tuple[str, str, str]]] = {
    "acm:PAP": [("paper", "written_by", "author"), ("author", "writes", "paper")],
    "acm:PSP": [("paper", "has_subject", "subject"), ("subject", "subject_of", "paper")],
    "acm:PTP": [("paper", "has_term", "term"), ("term", "term_in", "paper")],
    "dblp:APA": [("author", "writes", "paper"), ("paper", "written_by", "author")],
    "dblp:APVPA": [
        ("author", "writes", "paper"),
        ("paper", "published_in", "venue"),
        ("venue", "publishes", "paper"),
        ("paper", "written_by", "author"),
    ],
    "dblp:APTPA": [
        ("author", "writes", "paper"),
        ("paper", "has_term", "term"),
        ("term", "term_in", "paper"),
        ("paper", "written_by", "author"),
    ],
    "dblp:APCPA": [
        ("author", "writes", "paper"),
        ("paper", "published_in", "conference"),
        ("conference", "publishes", "paper"),
        ("paper", "written_by", "author"),
    ],
    "imdb:MAM": [("movie", "has_actor", "actor"), ("actor", "acts_in", "movie")],
    "imdb:MDM": [("movie", "directed_by", "director"), ("director", "directs", "movie")],
    "imdb:MKM": [("movie", "has_keyword", "keyword"), ("keyword", "keyword_in", "movie")],
}


DATASET_METAPATH_ORDER = {
    "acm": ["PAP", "PSP", "PTP"],
    "dblp": ["APA", "APVPA", "APTPA", "APCPA"],
    "imdb": ["MAM", "MDM", "MKM"],
}


def available_metapath_specs(
    dataset: str,
    relations: list[DirectedRelation],
    target_type: str,
) -> tuple[dict[str, list[DirectedRelation]], dict[str, str]]:
    dataset = dataset.lower()
    relation_lookup = {(r.source_type, r.relation_name, r.destination_type): r for r in relations}
    available: dict[str, list[DirectedRelation]] = {}
    skipped: dict[str, str] = {}
    for name in DATASET_METAPATH_ORDER.get(dataset, []):
        triples = METAPATH_NAMES[f"{dataset}:{name}"]
        if triples[0][0] != target_type or triples[-1][2] != target_type:
            skipped[name] = "target_type_mismatch"
            continue
        path: list[DirectedRelation] = []
        missing = False
        for triple in triples:
            relation = relation_lookup.get(triple)
            if relation is None:
                missing = True
                break
            path.append(relation)
        if missing:
            skipped[name] = "schema_missing"
        else:
            available[name] = path
    return available, skipped
