"""Training and scoring of the emulators.

The loss is the mean squared error of the predicted monthly change, computed
after dividing each target by its own standard deviation.  Without that scaling
the surface enthalpy change, which is tens of watt years per square metre, would
dominate the sub-halocline temperature change, which is a fraction of a kelvin,
and the network would learn only the first of the two.

Skill is reported as the fraction of the variance of the change that the
emulator explains, evaluated on ensemble members the model never saw.  The
comparison that matters is against persistence, meaning a prediction of no
change at all; an emulator that cannot beat it has learned nothing, however
small its error looks in absolute terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import SEED
from ..spatial.ensemble import EmulationDataset
from .gnn import GraphTensors, build, count_parameters


@dataclass
class TrainingHistory:
    """Loss and skill recorded while fitting."""

    epoch: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    test_loss: list = field(default_factory=list)


@dataclass
class Score:
    """Skill of one fitted emulator on held-out members."""

    architecture: str
    seed: int
    n_parameters: int
    variance_explained: np.ndarray      # per target
    rmse: np.ndarray                    # per target, in physical units
    rmse_persistence: np.ndarray        # the same for a no-change prediction
    thickness_variance_explained: float
    history: TrainingHistory

    def describe(self) -> str:
        return (f"{self.architecture:12s} seed {self.seed} "
                f"parameters {self.n_parameters:6d}  "
                f"variance explained "
                f"{self.variance_explained[0]:.3f} / "
                f"{self.variance_explained[1]:.3f}  "
                f"thickness {self.thickness_variance_explained:.3f}")


def set_threads(n_threads: int = 4) -> None:
    """Cap the processor threads used, so a laptop stays usable."""
    torch.set_num_threads(int(n_threads))


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def train_emulator(dataset: EmulationDataset, graph, architecture: str,
                   seed: int = SEED, n_epochs: int = 60, batch_size: int = 8,
                   learning_rate: float = 2e-3, weight_decay: float = 1e-5,
                   n_hidden: int = 48, n_layers: int = 3, n_heads: int = 4,
                   max_samples: int | None = None, device: str = "cpu",
                   verbose: bool = True) -> tuple[nn.Module, Score]:
    """Fit one emulator and score it on held-out ensemble members."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_index, test_index = dataset.split(seed=seed)
    if max_samples is not None and train_index.size > max_samples:
        train_index = rng.choice(train_index, max_samples, replace=False)

    # Scaling comes from the training split alone; using the whole record would
    # let information about the held-out members reach the fitted model.
    train_features = dataset.features[train_index]
    mean = train_features.reshape(-1, dataset.n_features).mean(0)
    std = train_features.reshape(-1, dataset.n_features).std(0) + 1e-6
    target_std = dataset.targets[train_index].reshape(-1, 2).std(0) + 1e-6

    def prepare(index):
        x = (dataset.features[index] - mean) / std
        y = dataset.targets[index] / target_std
        return (torch.as_tensor(x, dtype=torch.float32, device=device),
                torch.as_tensor(y, dtype=torch.float32, device=device))

    x_train, y_train = prepare(train_index)
    x_test, y_test = prepare(test_index)

    tensors = GraphTensors.from_graph(graph, device=device)
    model = build(architecture, dataset.n_features,
                  edge_dim=tensors.edge_features.shape[1],
                  n_hidden=n_hidden, n_layers=n_layers,
                  **({"n_heads": n_heads}
                     if architecture in ("attention", "edge") else {})).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, n_epochs)
    history = TrainingHistory()

    n_train = x_train.shape[0]
    for epoch in range(n_epochs):
        model.train()
        order = torch.randperm(n_train)
        total = 0.0
        for start in range(0, n_train, batch_size):
            batch = order[start:start + batch_size]
            optimiser.zero_grad()
            # Each sample is a whole field on the same graph, so the batch is
            # looped over rather than concatenated into one large graph.  With
            # a couple of thousand nodes the difference is negligible and the
            # attention coefficients stay easy to attribute to a single sample.
            loss = torch.zeros((), device=device)
            for i in batch:
                prediction = model(x_train[i], tensors)
                loss = loss + torch.mean((prediction - y_train[i]) ** 2)
            loss = loss / batch.numel()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss) * batch.numel()
        schedule.step()

        train_loss = total / n_train
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            test_loss = _evaluate_loss(model, x_test, y_test, tensors)
            history.epoch.append(epoch)
            history.train_loss.append(train_loss)
            history.test_loss.append(test_loss)
            print(f"    epoch {epoch:3d}  train {train_loss:.4f}  "
                  f"test {test_loss:.4f}")

    score = score_emulator(model, dataset, tensors, test_index, mean, std,
                           target_std, architecture, seed, history, device)
    return model, score


def _evaluate_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                   tensors: GraphTensors) -> float:
    model.eval()
    with torch.no_grad():
        total = sum(float(torch.mean((model(x[i], tensors) - y[i]) ** 2))
                    for i in range(x.shape[0]))
    return total / max(x.shape[0], 1)


def predict(model: nn.Module, dataset: EmulationDataset, tensors: GraphTensors,
            index: np.ndarray, mean: np.ndarray, std: np.ndarray,
            target_std: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Predicted change in physical units for the given samples."""
    model.eval()
    x = torch.as_tensor((dataset.features[index] - mean) / std,
                        dtype=torch.float32, device=device)
    with torch.no_grad():
        out = np.stack([model(x[i], tensors).cpu().numpy()
                        for i in range(x.shape[0])])
    return out * target_std


def score_emulator(model: nn.Module, dataset: EmulationDataset,
                   tensors: GraphTensors, test_index: np.ndarray,
                   mean: np.ndarray, std: np.ndarray, target_std: np.ndarray,
                   architecture: str, seed: int, history: TrainingHistory,
                   device: str = "cpu") -> Score:
    """Score an emulator against the held-out members."""
    from ..config import LATENT_ICE

    truth = dataset.targets[test_index]
    prediction = predict(model, dataset, tensors, test_index, mean, std,
                         target_std, device)

    residual = prediction - truth
    rmse = np.sqrt((residual ** 2).reshape(-1, 2).mean(0))
    rmse_persistence = np.sqrt((truth ** 2).reshape(-1, 2).mean(0))
    variance = truth.reshape(-1, 2).var(0)
    explained = 1.0 - (residual ** 2).reshape(-1, 2).mean(0) / variance

    # The change in enthalpy is what the network predicts; expressing its skill
    # as a thickness change makes it comparable with the observational record.
    thickness_truth = -truth[..., 0] / LATENT_ICE
    thickness_predicted = -prediction[..., 0] / LATENT_ICE
    thickness_explained = float(
        1.0 - np.mean((thickness_predicted - thickness_truth) ** 2)
        / np.var(thickness_truth))

    return Score(architecture=architecture, seed=seed,
                 n_parameters=count_parameters(model),
                 variance_explained=explained, rmse=rmse,
                 rmse_persistence=rmse_persistence,
                 thickness_variance_explained=thickness_explained,
                 history=history)


def save_model(model: nn.Module, path: Path, mean: np.ndarray,
               std: np.ndarray, target_std: np.ndarray) -> None:
    """Store weights together with the scaling needed to use them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_mean": mean,
                "feature_std": std, "target_std": target_std}, path)
