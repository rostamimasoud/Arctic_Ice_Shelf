"""Generation of the training ensemble and assembly of the learning problem.

The emulators are trained to take the state of the whole domain at the start of
a month, together with the forcing acting on it, and predict how that state
changes over the month.  Predicting the change instead of the new state is
deliberate.  Sea ice thickness in a given cell is very strongly correlated from
one month to the next, so a model asked for the new state can reach a high score
by returning its input almost unchanged, and the score would then measure
persistence instead of learned physics.

The ensemble spans a range of greenhouse forcings so that the emulator sees both
the present-day Arctic and states well beyond it.  Each member is spun up before
recording begins, and a small stochastic component is added to the surface
budget so that the record contains interannual variability for the early warning
analysis to work on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, SEED
from ..physics.seaice import LATENT_ICE, SeaIceParams
from .grid import ArcticGraph, ArcticGrid, build_graph, build_grid
from ..physics.insolation import surface_shortwave
from .model import SpatialArcticModel

#: Names of the per-cell input features, in the order they are assembled.
FEATURE_NAMES = ("ice_thickness", "open_fraction", "deep_temperature",
                 "shortwave", "shortwave_tendency", "latitude",
                 "forcing", "month_sine", "month_cosine")

#: Names of the predicted quantities.
TARGET_NAMES = ("enthalpy_change", "deep_temperature_change")

#: Snapshots stored per year.
MONTHS_PER_YEAR = 12


@dataclass
class EmulationDataset:
    """Inputs, targets and the metadata needed to interpret them."""

    features: np.ndarray        # (n_samples, n_cells, n_features), float32
    targets: np.ndarray         # (n_samples, n_cells, 2), float32
    forcing: np.ndarray         # (n_samples,)
    month: np.ndarray           # (n_samples,), 1 to 12
    member: np.ndarray          # (n_samples,), index of the ensemble member
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_std: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[2])

    def normalise(self, features: np.ndarray) -> np.ndarray:
        """Standardise features with the statistics of the training split."""
        return (features - self.feature_mean) / self.feature_std

    def split(self, fraction: float = 0.2, seed: int = SEED
              ) -> tuple[np.ndarray, np.ndarray]:
        """Split by ensemble member, returning training and test indices.

        Splitting by member rather than by sample is essential.  Consecutive
        months of one integration are nearly the same field, so a random split
        would place almost every test sample beside a training sample drawn from
        the same trajectory, and the reported skill would be inflated.
        """
        members = np.unique(self.member)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(members)
        n_test = max(1, int(round(fraction * members.size)))
        test_members = set(shuffled[:n_test].tolist())

        is_test = np.array([m in test_members for m in self.member])
        return np.where(~is_test)[0], np.where(is_test)[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, features=self.features, targets=self.targets,
            forcing=self.forcing, month=self.month, member=self.member,
            feature_mean=self.feature_mean, feature_std=self.feature_std,
            target_std=self.target_std)

    @classmethod
    def load(cls, path: Path) -> "EmulationDataset":
        stored = np.load(path)
        return cls(**{k: stored[k] for k in
                      ("features", "targets", "forcing", "month", "member",
                       "feature_mean", "feature_std", "target_std")})


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #

def assemble_features(model: SpatialArcticModel, grid: ArcticGrid,
                      state: np.ndarray, time_years: float,
                      forcing: float) -> np.ndarray:
    """Build the input features for one snapshot, shape ``(n_cells, n_features)``.

    The features are the ones a physical account of the problem would use: the
    current ice and ocean state, the radiation arriving now and whether it is
    growing or shrinking, position, and the forcing.  The rate of change of
    insolation is included because melt depends on where in the season the cell
    sits, and the same instantaneous sunlight means something different in
    spring and in autumn.
    """
    enthalpy = state[:, 0]
    deep = state[:, 1]

    shortwave = surface_shortwave(grid.latitude, np.array([time_years]))[0]
    ahead = surface_shortwave(grid.latitude,
                              np.array([time_years + 1.0 / 24.0]))[0]
    behind = surface_shortwave(grid.latitude,
                               np.array([time_years - 1.0 / 24.0]))[0]

    phase = 2.0 * np.pi * (time_years % 1.0)
    ones = np.ones_like(enthalpy)

    return np.stack([
        model.thickness(enthalpy),
        model.concentration(enthalpy),
        deep,
        shortwave,
        (ahead - behind) * 12.0,
        grid.latitude,
        ones * forcing,
        ones * np.sin(phase),
        ones * np.cos(phase),
    ], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #

def run_member(model: SpatialArcticModel, forcing: float, years: int,
               spin_up_years: float, steps_per_year: int, noise: float,
               seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one member, returning monthly states and their times."""
    warm = model.with_forcing(forcing)
    _, settled = warm.integrate(warm.initial_state(), years=spin_up_years,
                                steps_per_year=steps_per_year,
                                noise=noise, seed=seed)

    store_every = max(1, steps_per_year // MONTHS_PER_YEAR)
    times, states = warm.integrate(settled[-1], years=float(years),
                                   steps_per_year=steps_per_year,
                                   store_every=store_every, noise=noise,
                                   seed=seed + 1)
    return times, states


def build_dataset(forcings: np.ndarray, years: int = 30,
                  spin_up_years: float = 90.0, steps_per_year: int = 192,
                  noise: float = 1.2, spacing_km: float = 100.0,
                  column: SeaIceParams | None = None,
                  seed: int = SEED, verbose: bool = True
                  ) -> tuple[EmulationDataset, ArcticGrid, ArcticGraph]:
    """Run the ensemble and assemble the learning problem.

    ``steps_per_year`` is a multiple of twelve so that stored snapshots fall on
    month boundaries exactly, which keeps the prediction interval identical for
    every sample.
    """
    grid = build_grid(spacing_km)
    graph = build_graph(grid)
    model = SpatialArcticModel(grid, graph, column=column)

    all_features = []
    all_targets = []
    all_forcing = []
    all_month = []
    all_member = []

    for member, forcing in enumerate(np.asarray(forcings, float)):
        times, states = run_member(model, float(forcing), years, spin_up_years,
                                   steps_per_year, noise, seed + 100 * member)
        if verbose:
            thickness = model.thickness(states[:, :, 0])
            print(f"  member {member:2d}  forcing {forcing:5.2f} W m-2  "
                  f"mean thickness {thickness.mean():.2f} m  "
                  f"extent {model.extent_km2(states[:, :, 0]).min() / 1e6:.2f} "
                  f"to {model.extent_km2(states[:, :, 0]).max() / 1e6:.2f} "
                  f"million km2")

        for i in range(states.shape[0] - 1):
            all_features.append(assemble_features(model, grid, states[i],
                                                  float(times[i]), float(forcing)))
            all_targets.append((states[i + 1] - states[i]).astype(np.float32))
            all_forcing.append(float(forcing))
            all_month.append(int(round((times[i] % 1.0) * 12)) % 12 + 1)
            all_member.append(member)

    features = np.stack(all_features)
    targets = np.stack(all_targets)

    # Scaling statistics are computed over everything here; the training script
    # recomputes them on the training split alone before fitting.
    feature_mean = features.reshape(-1, features.shape[-1]).mean(0)
    feature_std = features.reshape(-1, features.shape[-1]).std(0) + 1e-6
    target_std = targets.reshape(-1, targets.shape[-1]).std(0) + 1e-6

    dataset = EmulationDataset(
        features=features, targets=targets,
        forcing=np.array(all_forcing), month=np.array(all_month),
        member=np.array(all_member), feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_std=target_std.astype(np.float32))
    return dataset, grid, graph


def default_path(spacing_km: float = 100.0) -> Path:
    return DATA_DIR / f"emulation_dataset_{int(spacing_km)}km.npz"
