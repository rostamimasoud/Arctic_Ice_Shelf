"""How far influence travels, measured three ways.

The question the emulators are meant to answer is over what distance conditions
in one part of the Arctic control what happens in another, and how that distance
changes with season and with warming.  An attention weight on its own cannot
answer it, because nothing says a large attention weight corresponds to a large
physical influence.  This module therefore measures the same quantity three
times and compares them.

``physical``
    Perturb the state of the spatial model at one cell, integrate for a month,
    and see where the response appears.  This is a column of the Jacobian of the
    one-month map, obtained by finite differences.  Because the model's
    transport parameters were prescribed, this is the answer against which the
    other two are judged.

``emulator``
    Exactly the same protocol applied to a trained emulator instead of to the
    model.  Agreement with the physical measurement shows the emulator has
    learned the propagation of influence and did not merely fit local physics.

``attention``
    The composition of the learned attention matrices over all message passing
    rounds.  A single layer only reaches immediate neighbours, so the reach of
    the network is the product of its layers, and reading one layer alone would
    understate it by the number of rounds.

All three reduce to one number per source cell: the mean distance to which
influence spreads, weighted by how strongly each cell responds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import SEED
from ..models.gnn import GraphAttention, GraphTensors

#: Size of the enthalpy perturbation used for the finite differences,
#: W yr m^-2.  It corresponds to roughly five centimetres of ice: large enough
#: to lift the response above rounding, small enough to stay in the linear
#: regime where a Jacobian is meaningful.
PERTURBATION = 0.5

#: Responses weaker than this fraction of the strongest response are treated as
#: absent.  Without a floor the length scale is dominated by a long tail of
#: values at the level of rounding error, which sits at large distance simply
#: because most of the domain is far away.
RESPONSE_FLOOR = 1e-3


@dataclass
class LengthScale:
    """Influence length scales measured from one source cell each."""

    source: np.ndarray            # indices of the cells perturbed
    length_km: np.ndarray         # mean influence distance from each source
    response: np.ndarray          # (n_source, n_cells) normalised response
    method: str
    label: str = ""

    def summary(self) -> tuple:
        """Median length scale and its interquartile range, km."""
        good = np.isfinite(self.length_km)
        if not good.any():
            return float("nan"), float("nan"), float("nan")
        values = self.length_km[good]
        return (float(np.median(values)),
                float(np.percentile(values, 25)),
                float(np.percentile(values, 75)))


def _weighted_distance(response: np.ndarray, distance: np.ndarray
                       ) -> np.ndarray:
    """Mean distance of the response from its source, one value per source."""
    weight = np.abs(response)
    peak = weight.max(axis=1, keepdims=True)
    weight = np.where(weight >= RESPONSE_FLOOR * np.maximum(peak, 1e-30),
                      weight, 0.0)
    total = weight.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (weight * distance).sum(axis=1) / total,
                        np.nan)


def choose_sources(grid, n_sources: int = 64, seed: int = SEED,
                   min_latitude: float = 68.0) -> np.ndarray:
    """Pick source cells spread over the deep basin.

    Cells close to the southern edge of the domain are excluded, because a
    response there runs out of the domain rather than spreading, and the
    truncated response would be recorded as a short length scale that says more
    about the domain boundary than about the physics.
    """
    eligible = np.where(grid.latitude >= min_latitude)[0]
    rng = np.random.default_rng(seed)
    if eligible.size <= n_sources:
        return eligible
    return np.sort(rng.choice(eligible, n_sources, replace=False))


# --------------------------------------------------------------------------- #
# The physical measurement
# --------------------------------------------------------------------------- #

def physical_length_scale(model, grid, state: np.ndarray,
                          sources: np.ndarray, months: float = 1.0,
                          steps_per_year: int = 192, t0: float = 0.0,
                          perturbation: float = PERTURBATION,
                          label: str = "") -> LengthScale:
    """Influence length scale of the spatial model itself.

    One cell is nudged, the model is advanced for ``months``, and the difference
    from an unperturbed run of the same length is recorded everywhere.
    """
    years = months / 12.0
    _, control = model.integrate(state, years=years,
                                 steps_per_year=steps_per_year, t0=t0)
    reference = control[-1]

    distance = np.hypot(grid.x[:, None] - grid.x[None, :],
                        grid.y[:, None] - grid.y[None, :])

    response = np.empty((sources.size, grid.n_cells))
    for k, cell in enumerate(sources):
        nudged = state.copy()
        nudged[cell, 0] += perturbation
        _, evolved = model.integrate(nudged, years=years,
                                     steps_per_year=steps_per_year, t0=t0)
        response[k] = (evolved[-1][:, 0] - reference[:, 0]) / perturbation

    return LengthScale(source=sources,
                       length_km=_weighted_distance(response,
                                                    distance[sources]),
                       response=response, method="physical", label=label)


# --------------------------------------------------------------------------- #
# The emulator measurement
# --------------------------------------------------------------------------- #

def emulator_length_scale(model, tensors: GraphTensors, grid,
                          features: np.ndarray, mean: np.ndarray,
                          std: np.ndarray, sources: np.ndarray,
                          feature_index: int = 0,
                          perturbation: float = 0.05,
                          label: str = "") -> LengthScale:
    """Influence length scale of a trained emulator, by the same protocol.

    The perturbation is applied to the standardised input, so its size is in
    units of the spread of that feature across the training set.
    """
    model.eval()
    distance = np.hypot(grid.x[:, None] - grid.x[None, :],
                        grid.y[:, None] - grid.y[None, :])

    scaled = (features - mean) / std
    base_input = torch.as_tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        reference = model(base_input, tensors).numpy()[:, 0]

    response = np.empty((sources.size, grid.n_cells))
    for k, cell in enumerate(sources):
        nudged = base_input.clone()
        nudged[cell, feature_index] += perturbation
        with torch.no_grad():
            evolved = model(nudged, tensors).numpy()[:, 0]
        response[k] = (evolved - reference) / perturbation

    return LengthScale(source=sources,
                       length_km=_weighted_distance(response,
                                                    distance[sources]),
                       response=response, method="emulator", label=label)


# --------------------------------------------------------------------------- #
# The attention measurement
# --------------------------------------------------------------------------- #

def composed_attention(model, tensors: GraphTensors,
                       features: torch.Tensor) -> np.ndarray:
    """Influence implied by the attention weights of every layer combined.

    Each layer's coefficients form a sparse row-stochastic matrix on the graph.
    A single layer reaches only immediate neighbours, so the reach of the whole
    network is the matrix product across layers.  Returns a dense matrix whose
    entry ``(i, j)`` is the weight with which cell ``j`` reaches cell ``i``.
    """
    from scipy import sparse

    model.eval()
    with torch.no_grad():
        model(features, tensors)

    layers = [layer for layer in model.modules()
              if isinstance(layer, GraphAttention) and layer.attention is not None]
    if not layers:
        raise RuntimeError(
            "this emulator has no attention layers, so there are no "
            "coefficients to compose")

    source, target = tensors.edge_index.numpy()
    n = tensors.n_nodes

    combined = None
    for layer in layers:
        # Heads are averaged: they are parallel views of the same neighbourhood
        # and the layer's output sums them, so their mean is the weight with
        # which a neighbour reaches the node.
        weight = layer.attention.mean(dim=1).numpy()
        matrix = sparse.coo_matrix((weight, (target, source)),
                                   shape=(n, n)).tocsr()
        combined = matrix if combined is None else matrix.dot(combined)

    return np.asarray(combined.todense())


def attention_length_scale(model, tensors: GraphTensors, grid,
                           features: np.ndarray, mean: np.ndarray,
                           std: np.ndarray, sources: np.ndarray,
                           label: str = "") -> LengthScale:
    """Influence length scale read from the composed attention weights."""
    scaled = torch.as_tensor((features - mean) / std, dtype=torch.float32)
    influence = composed_attention(model, tensors, scaled)

    distance = np.hypot(grid.x[:, None] - grid.x[None, :],
                        grid.y[:, None] - grid.y[None, :])
    # Column j of the influence matrix is how far cell j reaches, which is the
    # same convention as the perturbation experiments.
    response = influence[:, sources].T

    return LengthScale(source=sources,
                       length_km=_weighted_distance(response,
                                                    distance[sources]),
                       response=response, method="attention", label=label)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

@dataclass
class Comparison:
    """Agreement between two sets of length scales over the same sources."""

    correlation: float
    slope: float
    bias_km: float
    n: int


def compare(reference: LengthScale, estimate: LengthScale) -> Comparison:
    """Compare two length scale measurements made from the same source cells."""
    if not np.array_equal(reference.source, estimate.source):
        raise ValueError("the two measurements used different source cells")

    good = np.isfinite(reference.length_km) & np.isfinite(estimate.length_km)
    a, b = reference.length_km[good], estimate.length_km[good]
    if a.size < 3:
        return Comparison(float("nan"), float("nan"), float("nan"), int(a.size))

    slope = float(np.polyfit(a, b, 1)[0])
    return Comparison(correlation=float(np.corrcoef(a, b)[0, 1]),
                      slope=slope, bias_km=float(np.mean(b - a)), n=int(a.size))
