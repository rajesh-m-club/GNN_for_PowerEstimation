"""
model/model.py

Edge-Aware Attentive GraphSAGE model for circuit activity prediction.

This module ONLY defines the model. It does not touch the dataset
builder, netlist parser, SAIF parser, or feature builder, and makes no
assumptions about how many circuits, nodes, or edges exist -- all shapes
are read dynamically from the PyG `Data` object at forward time.

------------------------------------------------------------------
Tensor shapes through the network (num_nodes = N, num_edges = E)
------------------------------------------------------------------
data.x           : [N, 5]     -> node_feat_attn -> [N, 5]  (re-weighted)
                               -> node_encoder   -> [N, hidden_dim]
data.edge_attr   : [E, 17]    -> edge_feat_attn -> [E, 17] (re-weighted)
                               -> edge_encoder   -> [E, hidden_dim]

per EdgeAwareSAGELayer (repeated num_layers times):
    h_u = h[src]                         [E, hidden_dim]
    h_v = h[dst]                         [E, hidden_dim]
    message_uv = MLP([h_u ; e_uv])       [E, hidden_dim]
    score_uv   = MLP([h_u ; e_uv ; h_v]) [E, 1] -> squeeze -> [E]
    alpha_uv   = softmax(score_uv) grouped by dst
    agg_v      = scatter_sum(alpha_uv * message_uv, dst)  [N, hidden_dim]
    h_v_next   = MLP([h_v ; agg_v]) + h_v (residual)       [N, hidden_dim]

final hidden state h : [N, hidden_dim]
    -> p0_head    -> [N, 1]
    -> p1_head    -> [N, 1]
    -> psw01_head -> [N, 1]
    -> psw10_head -> [N, 1]
    concatenated  -> [N, 4]  in the fixed order [p0, p1, psw01, psw10]

------------------------------------------------------------------
Directed-edge handling
------------------------------------------------------------------
`edge_index[0]` is always treated as the source node (u) and
`edge_index[1]` as the destination node (v). Messages are built from the
source representation and the edge representation, and are aggregated
(and softmax-normalized) per destination node. The graph is never
symmetrized/converted to undirected anywhere in this file.

------------------------------------------------------------------
17-D edge feature usage
------------------------------------------------------------------
`edge_attr` is passed through a learned per-channel gate
(`edge_feat_attn`), encoded into a `hidden_dim`-wide edge representation
(`edge_encoder`), and that encoded edge representation is concatenated
into BOTH the message MLP and the attention MLP inside every
`EdgeAwareSAGELayer`. Edge information is therefore an explicit input to
message construction and to neighbor-attention scoring, not just a
side channel.

------------------------------------------------------------------
Output constraints (see item 18 in the task spec)
------------------------------------------------------------------
[p0, p1, psw01, psw10] are NOT four mutually exclusive classes, so a
softmax across the 4 outputs would be wrong (e.g. p0 and p1 alone
already sum to 1, but psw01/psw10 are a separate pair of switching
probabilities with their own, generally different, scale). Instead:

  * each output head is followed by an independent `torch.sigmoid`
    (controlled by `constrain_output`, default True), which keeps every
    output in the valid [0, 1] probability range without coupling the
    four values together, and
  * if `constrain_output=False`, the heads emit raw continuous values
    and any range constraint is left entirely to the loss/regularizer
    (e.g. a penalty term added in train.py), which is useful if you
    want to debug whether the sigmoid is limiting fit quality.

------------------------------------------------------------------
target_mask
------------------------------------------------------------------
This model does not consume or alter `data.target_mask` -- masking is a
loss/evaluation-time concern, not a forward-pass concern, so it is
applied in `train.py` / `evaluate.py` (`pred[mask]`, `data.y[mask]`).

Note on what target_mask actually selects (set by graph/feature_builder.py):
primary-input nodes get their `x` from real SAIF-measured activity (the
known input stimulus) and are therefore EXCLUDED from target_mask --
they are given, not predicted. Only internal (non-PI) nets, whose
activity actually needs to be inferred from the graph, are included as
targets. The model still runs its forward pass over every node
(including PIs, since they're needed as message-passing inputs); only
the loss/metric computation ignores their output.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from utilis.attention import FeatureAttention, EdgeAwareSAGELayer


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class EdgeAwareGraphSAGE(nn.Module):
    """
    Edge-Aware Attentive GraphSAGE for per-node circuit activity
    prediction.

    Args:
        node_in_dim: number of input node features (fixed at 5:
            [p0, p1, pt1, pt0, inv]).
        edge_in_dim: number of input edge features (fixed at 17:
            [SC, TC_00_00, ..., TC_11_11]).
        hidden_dim: width of node/edge encodings and message-passing
            layers.
        num_layers: number of stacked EdgeAwareSAGELayer blocks.
        dropout: dropout used inside the update MLP and output heads.
        head_hidden_dim: hidden width of each of the 4 output heads.
        constrain_output: if True (default), apply an independent
            sigmoid to each of the 4 outputs (see module docstring).
    """

    NODE_FEATURE_NAMES = ["p0", "p1", "pt1", "pt0", "inv"]
    EDGE_FEATURE_NAMES = [
        "SC",
        "TC_00_00", "TC_00_01", "TC_00_10", "TC_00_11",
        "TC_01_00", "TC_01_01", "TC_01_10", "TC_01_11",
        "TC_10_00", "TC_10_01", "TC_10_10", "TC_10_11",
        "TC_11_00", "TC_11_01", "TC_11_10", "TC_11_11",
    ]
    TARGET_NAMES = ["p0", "p1", "psw01", "psw10"]

    def __init__(
        self,
        node_in_dim: int = 5,
        edge_in_dim: int = 17,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_dim: int = 32,
        constrain_output: bool = True,
    ):
        super().__init__()
        assert node_in_dim == len(self.NODE_FEATURE_NAMES), (
            "node_in_dim must match the fixed [p0, p1, pt1, pt0, inv] layout"
        )
        assert edge_in_dim == len(self.EDGE_FEATURE_NAMES), (
            "edge_in_dim must match the fixed 17-D SC/TC edge layout"
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.constrain_output = constrain_output

        # --- feature gating (interpretable, learned, per-channel) ---
        self.node_feat_attn = FeatureAttention(node_in_dim)
        self.edge_feat_attn = FeatureAttention(edge_in_dim)

        # --- encoders ---
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim),
        )

        # --- edge-aware attentive GraphSAGE stack ---
        self.layers = nn.ModuleList(
            [EdgeAwareSAGELayer(hidden_dim, dropout) for _ in range(num_layers)]
        )

        # --- target-specific output heads ---
        self.p0_head = _mlp(hidden_dim, head_hidden_dim, 1, dropout)
        self.p1_head = _mlp(hidden_dim, head_hidden_dim, 1, dropout)
        self.psw01_head = _mlp(hidden_dim, head_hidden_dim, 1, dropout)
        self.psw10_head = _mlp(hidden_dim, head_hidden_dim, 1, dropout)

    def forward(
        self, data, return_attention: bool = False
    ):
        """
        Args:
            data: a PyG `Data` (or batched `Batch`) object with
                data.x          [N, 5]
                data.edge_index [2, E]
                data.edge_attr  [E, 17]
            return_attention: if True, also return a dict of
                interpretability tensors (see below).

        Returns:
            pred: [N, 4] tensor, columns ordered
                  [p0, p1, psw01, psw10].
            attention (only if return_attention=True): dict with
                - "node_feature_attention": [N, 5]   (per-node softmax
                  weights over [p0, p1, pt1, pt0, inv])
                - "edge_feature_attention": [E, 17]  (per-edge softmax
                  weights over the 17 SC/TC channels)
                - "neighbor_attention": List[Tensor], one [E] tensor per
                  message-passing layer, aligned with data.edge_index
                  (i.e. neighbor_attention[l][k] is alpha for the edge
                  edge_index[:, k] at layer l). These are learned
                  attention/gating values, not a claim of causal
                  importance.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x_weighted, node_alpha = self.node_feat_attn(x)
        e_weighted, edge_alpha = self.edge_feat_attn(edge_attr)

        h = self.node_encoder(x_weighted)   # [N, hidden_dim]
        e = self.edge_encoder(e_weighted)   # [E, hidden_dim]  (fixed across layers)

        neighbor_attentions: List[torch.Tensor] = []
        for layer in self.layers:
            h, alpha = layer(h, edge_index, e, return_attention=return_attention)
            if return_attention:
                neighbor_attentions.append(alpha)

        p0 = self.p0_head(h)
        p1 = self.p1_head(h)
        psw01 = self.psw01_head(h)
        psw10 = self.psw10_head(h)

        pred = torch.cat([p0, p1, psw01, psw10], dim=-1)  # [N, 4]
        if self.constrain_output:
            pred = torch.sigmoid(pred)

        if return_attention:
            attention: Dict[str, object] = {
                "node_feature_attention": node_alpha,
                "edge_feature_attention": edge_alpha,
                "neighbor_attention": neighbor_attentions,
            }
            return pred, attention
        return pred