from __future__ import annotations

from dataclasses import dataclass

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.metapath_blocks import two_hop_block_name


@dataclass(frozen=True)
class MetaPathSchemaDefaults:
    dataset: str
    target_type: str
    requested_blocks: list[str]
    available_blocks: list[str]
    skipped_blocks: list[str]
    required_blocks: list[str]
    relation_by_block: dict[str, DirectedRelation]


DEFAULT_REQUESTED: dict[str, list[str]] = {
    "acm": ["PAP", "PSP", "PTP"],
    "dblp": ["APA", "APVPA", "APTPA", "APCPA"],
    "imdb": ["MAM", "MDM", "MKM"],
}

REQUIRED_BLOCKS: dict[str, list[str]] = {
    "acm": ["PAP"],
    "dblp": ["APA"],
    "imdb": ["MAM", "MDM", "MKM"],
}


def schema_default_metapath_blocks(
    *,
    dataset_name: str,
    target_type: str,
    relations: list[DirectedRelation],
    requested_blocks: list[str] | None = None,
) -> MetaPathSchemaDefaults:
    dataset = dataset_name.lower()
    requested = [name.upper() for name in (requested_blocks or DEFAULT_REQUESTED.get(dataset, []))]
    relation_by_block: dict[str, DirectedRelation] = {}
    for relation in relations:
        if relation.destination_type != target_type or relation.source_type == target_type:
            continue
        relation_by_block.setdefault(two_hop_block_name(target_type, relation.source_type), relation)
    available = [name for name in requested if name in relation_by_block]
    skipped = [name for name in requested if name not in relation_by_block]
    required = REQUIRED_BLOCKS.get(dataset, [])
    return MetaPathSchemaDefaults(
        dataset=dataset,
        target_type=target_type,
        requested_blocks=requested,
        available_blocks=available,
        skipped_blocks=skipped,
        required_blocks=required,
        relation_by_block={name: relation_by_block[name] for name in available},
    )


def default_metapath_blocks(dataset_name: str, target_type: str, relations: list[DirectedRelation]) -> list[str]:
    return schema_default_metapath_blocks(
        dataset_name=dataset_name,
        target_type=target_type,
        relations=relations,
    ).available_blocks

