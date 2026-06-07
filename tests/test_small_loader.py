from shadow_hgc.data.small import load_processed_small_dataset


def test_processed_small_loader_keeps_only_incoming_relations_to_target():
    for dataset in ["acm", "dblp", "imdb"]:
        graph = load_processed_small_dataset(dataset)
        assert graph.target_type in graph.node_features
        assert graph.train_idx.numel() > 0
        assert graph.test_idx.numel() > 0
        assert graph.relations
        assert all(relation.destination_type == graph.target_type for relation in graph.relations)
        assert all(relation in graph.edge_index for relation in graph.relations)
