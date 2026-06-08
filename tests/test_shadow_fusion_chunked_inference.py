import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.models.shadow_fusion import ShadowFusionClassifier


def test_shadow_fusion_chunked_target_inference_matches_full_forward_on_toy_graph():
    relation = DirectedRelation("author", "writes", "paper")
    torch.manual_seed(7)
    model = ShadowFusionClassifier(
        in_channels={"author": 3, "paper": 2},
        out_channels=4,
        node_types=["author", "paper"],
        relations=[relation],
        target_type="paper",
        block_in_channels={"diffusion:X1": 2},
        hidden_dim=5,
        dropout=0.0,
    )
    model.eval()
    x_dict = {
        "author": torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10.0,
        "paper": torch.arange(10, dtype=torch.float32).reshape(5, 2) / 7.0,
    }
    edge_index = {relation: torch.tensor([[0, 1, 2, 3, 4, 5, 0], [0, 1, 1, 2, 3, 3, 4]], dtype=torch.long)}
    edge_weight = {relation: torch.tensor([1.0, 0.25, 0.75, 1.0, 0.4, 0.6, 1.0])}
    blocks = {"paper": {"diffusion:X1": torch.randn(5, 2)}}

    full = model(x_dict, edge_index, edge_weight, block_feature_dict=blocks)["paper"]
    chunked = model.infer_target_chunked(
        x_dict,
        edge_index,
        edge_weight,
        block_feature_dict=blocks,
        dst_chunk_size=2,
    )

    assert torch.allclose(chunked, full, atol=1e-6)
