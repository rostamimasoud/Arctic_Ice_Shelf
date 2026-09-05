"""Emulator architectures.

Four models are implemented, sharing one interface so that they can be trained
and scored by the same code:

``DenseEmulator``
    A baseline that sees each cell on its own, with no knowledge of its
    neighbours.  It is the control that shows how much of the emulator's skill
    comes from spatial information rather than from local physics.
``GraphConvolutionEmulator``
    Message passing with fixed weights set by the graph, following the standard
    symmetric normalisation.
``GraphAttentionEmulator``
    Message passing with weights the network learns for every edge.  Those
    weights are the object the connectivity analysis reads: they say how much
    each neighbour mattered for each prediction.
``EdgeConditionedEmulator``
    Attention weights conditioned additionally on the geometry of the edge, so
    that direction and distance can modulate influence directly.

The layers are written out rather than taken from a graph learning library.
The reason is not preference: the analysis needs the attention coefficients
themselves, per edge and per layer, and needs them to be exactly the numbers
used in the forward pass.  Writing the propagation explicitly makes that
inspection unambiguous and removes a heavy dependency.

Message passing is expressed with index arithmetic.  For each edge the source
node's features are gathered, transformed, weighted, and added into the target
node with a scatter operation.  With roughly eighteen thousand edges this is
comfortably fast on a processor and needs no sparse matrix support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def scatter_sum(values: torch.Tensor, index: torch.Tensor,
                n_nodes: int) -> torch.Tensor:
    """Sum ``values`` into ``n_nodes`` rows according to ``index``."""
    shape = (n_nodes,) + values.shape[1:]
    out = torch.zeros(shape, dtype=values.dtype, device=values.device)
    expanded = index.view(-1, *([1] * (values.dim() - 1))).expand_as(values)
    return out.scatter_add_(0, expanded, values)


def scatter_softmax(values: torch.Tensor, index: torch.Tensor,
                    n_nodes: int) -> torch.Tensor:
    """Softmax over the edges sharing each target node.

    Subtracting the per-node maximum before exponentiating keeps the result
    finite when the raw scores are large, which they become once the network has
    learned to concentrate on a few neighbours.
    """
    shape = (n_nodes,) + values.shape[1:]
    largest = torch.full(shape, float("-inf"), dtype=values.dtype,
                         device=values.device)
    expanded = index.view(-1, *([1] * (values.dim() - 1))).expand_as(values)
    largest = largest.scatter_reduce(0, expanded, values, reduce="amax",
                                     include_self=True)
    shifted = torch.exp(values - largest.gather(0, expanded))
    total = scatter_sum(shifted, index, n_nodes).gather(0, expanded)
    return shifted / (total + 1e-16)


@dataclass
class GraphTensors:
    """The graph as tensors, built once and reused by every layer."""

    edge_index: torch.Tensor      # (2, n_edges), long
    edge_features: torch.Tensor   # (n_edges, n_edge_features)
    n_nodes: int
    normalisation: torch.Tensor   # (n_edges,), symmetric degree normalisation

    @classmethod
    def from_graph(cls, graph, device: str = "cpu") -> "GraphTensors":
        edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long,
                                     device=device)
        features = torch.as_tensor(np.asarray(graph.edge_features,
                                              dtype=np.float32), device=device)
        degree = torch.zeros(graph.n_nodes, device=device)
        degree.scatter_add_(0, edge_index[1],
                            torch.ones(edge_index.shape[1], device=device))
        degree = degree.clamp(min=1.0)
        normalisation = (degree[edge_index[0]].rsqrt()
                         * degree[edge_index[1]].rsqrt())
        return cls(edge_index=edge_index, edge_features=features,
                   n_nodes=int(graph.n_nodes), normalisation=normalisation)


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #

class GraphConvolution(nn.Module):
    """Message passing with weights fixed by the graph."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor, graph: GraphTensors) -> torch.Tensor:
        source, target = graph.edge_index
        messages = self.linear(x)[source] * graph.normalisation.unsqueeze(-1)
        return scatter_sum(messages, target, graph.n_nodes)


class GraphAttention(nn.Module):
    """Message passing with learned per-edge weights.

    Attention is computed in the additive form: a score is formed for every edge
    from the transformed features of its two endpoints, and the scores of all
    edges arriving at a node are normalised to sum to one.  The normalised
    coefficients are retained after each forward pass so that the connectivity
    analysis can read them.
    """

    def __init__(self, n_in: int, n_out: int, n_heads: int = 4,
                 slope: float = 0.2, edge_dim: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_heads * n_out, bias=False)
        self.score_source = nn.Parameter(torch.empty(n_heads, n_out))
        self.score_target = nn.Parameter(torch.empty(n_heads, n_out))
        self.activation = nn.LeakyReLU(slope)
        self.bias = nn.Parameter(torch.zeros(n_heads * n_out))

        self.edge_dim = edge_dim
        if edge_dim is not None:
            # Conditioning the score on edge geometry lets direction and
            # distance modulate influence directly instead of only through the
            # node features.
            self.edge_linear = nn.Linear(edge_dim, n_heads * n_out, bias=False)
            self.score_edge = nn.Parameter(torch.empty(n_heads, n_out))
            nn.init.xavier_uniform_(self.score_edge)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.score_source)
        nn.init.xavier_uniform_(self.score_target)

        #: Attention coefficients from the most recent forward pass,
        #: shape ``(n_edges, n_heads)``.
        self.attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, graph: GraphTensors) -> torch.Tensor:
        source, target = graph.edge_index
        n_edges = source.shape[0]

        features = self.linear(x).view(-1, self.n_heads, self.n_out)
        score = ((features * self.score_source).sum(-1)[source]
                 + (features * self.score_target).sum(-1)[target])

        message = features[source]
        if self.edge_dim is not None:
            edge = self.edge_linear(graph.edge_features).view(
                n_edges, self.n_heads, self.n_out)
            score = score + (edge * self.score_edge).sum(-1)
            message = message + edge

        weight = scatter_softmax(self.activation(score), target, graph.n_nodes)
        self.attention = weight.detach()

        weighted = message * weight.unsqueeze(-1)
        gathered = scatter_sum(weighted, target, graph.n_nodes)
        return gathered.reshape(-1, self.n_heads * self.n_out) + self.bias


# --------------------------------------------------------------------------- #
# Emulators
# --------------------------------------------------------------------------- #

class BaseEmulator(nn.Module):
    """Shared machinery: what the emulator predicts and how it is scored.

    Every emulator maps the state and forcing of the whole domain at one time to
    the change in state over the following month.  Predicting the increment
    rather than the new state matters: the state is strongly autocorrelated from
    one month to the next, so a model predicting it directly can score well by
    copying its input, and the skill would say nothing about whether the physics
    was learned.
    """

    def __init__(self, n_features: int, n_targets: int = 2):
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets

    def attention_weights(self) -> list:
        """Attention from the most recent forward pass, one entry per layer."""
        return [layer.attention for layer in self.modules()
                if isinstance(layer, GraphAttention) and layer.attention is not None]


class DenseEmulator(BaseEmulator):
    """Baseline that treats every cell independently."""

    def __init__(self, n_features: int, n_hidden: int = 96, n_layers: int = 4,
                 n_targets: int = 2):
        super().__init__(n_features, n_targets)
        layers: list = []
        width = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(width, n_hidden), nn.SiLU()]
            width = n_hidden
        layers.append(nn.Linear(width, n_targets))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, graph: GraphTensors) -> torch.Tensor:
        return self.network(x)


class GraphConvolutionEmulator(BaseEmulator):
    """Message passing with fixed graph weights."""

    def __init__(self, n_features: int, n_hidden: int = 64, n_layers: int = 3,
                 n_targets: int = 2):
        super().__init__(n_features, n_targets)
        self.encoder = nn.Sequential(nn.Linear(n_features, n_hidden), nn.SiLU())
        self.blocks = nn.ModuleList(
            [GraphConvolution(n_hidden, n_hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(n_hidden) for _ in range(n_layers)])
        self.decoder = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(),
                                     nn.Linear(n_hidden, n_targets))

    def forward(self, x: torch.Tensor, graph: GraphTensors) -> torch.Tensor:
        h = self.encoder(x)
        for block, norm in zip(self.blocks, self.norms):
            # Residual connections keep the deeper stacks trainable and, more
            # importantly here, stop the node features from being smoothed into
            # one another, which would erase the spatial structure the
            # connectivity analysis is looking for.
            h = h + torch.nn.functional.silu(norm(block(h, graph)))
        return self.decoder(h)


class GraphAttentionEmulator(BaseEmulator):
    """Message passing with learned per-edge weights."""

    def __init__(self, n_features: int, n_hidden: int = 64, n_layers: int = 3,
                 n_heads: int = 4, n_targets: int = 2,
                 edge_dim: int | None = None):
        super().__init__(n_features, n_targets)
        if n_hidden % n_heads:
            raise ValueError("n_hidden must be divisible by n_heads")
        per_head = n_hidden // n_heads

        self.encoder = nn.Sequential(nn.Linear(n_features, n_hidden), nn.SiLU())
        self.blocks = nn.ModuleList([
            GraphAttention(n_hidden, per_head, n_heads=n_heads,
                           edge_dim=edge_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(n_hidden) for _ in range(n_layers)])
        self.decoder = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(),
                                     nn.Linear(n_hidden, n_targets))

    def forward(self, x: torch.Tensor, graph: GraphTensors) -> torch.Tensor:
        h = self.encoder(x)
        for block, norm in zip(self.blocks, self.norms):
            h = h + torch.nn.functional.silu(norm(block(h, graph)))
        return self.decoder(h)


class EdgeConditionedEmulator(GraphAttentionEmulator):
    """Attention additionally conditioned on the geometry of each edge."""

    def __init__(self, n_features: int, edge_dim: int, n_hidden: int = 64,
                 n_layers: int = 3, n_heads: int = 4, n_targets: int = 2):
        super().__init__(n_features, n_hidden=n_hidden, n_layers=n_layers,
                         n_heads=n_heads, n_targets=n_targets,
                         edge_dim=edge_dim)


#: Registry used by the training script.
ARCHITECTURES = {
    "dense": DenseEmulator,
    "convolution": GraphConvolutionEmulator,
    "attention": GraphAttentionEmulator,
    "edge": EdgeConditionedEmulator,
}


def build(architecture: str, n_features: int, edge_dim: int,
          **kwargs) -> BaseEmulator:
    """Construct one of the registered architectures."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of "
                         f"{sorted(ARCHITECTURES)}, got {architecture!r}")
    if architecture == "edge":
        return EdgeConditionedEmulator(n_features, edge_dim, **kwargs)
    return ARCHITECTURES[architecture](n_features, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
