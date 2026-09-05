"""Continuation problems built from the physical models.

Each class here wraps a model as a one-year map so that
:mod:`arcticgnn.dynsys.continuation` can trace its branches.  The wrapper is
thin on purpose: it exists only to declare the control parameter, the scales
that make the arclength norm well conditioned, and the diagnostic that is
plotted against the control parameter.
"""

from __future__ import annotations

import numpy as np

from ..physics.seaice import LATENT_ICE, SeaIceColumn, SeaIceParams
from .continuation import PeriodicProblem

#: Diagnostics that :class:`SeaIceProblem` can report.
OBSERVABLES = ("thickness_min", "thickness_max", "thickness_mean",
               "open_fraction", "deep_temperature")


class SeaIceProblem(PeriodicProblem):
    """The seasonal sea-ice column as a one-year map.

    The state is scaled by the latent heat of a metre of ice and by one kelvin,
    so that a unit change in either scaled variable is a comparable physical
    change.  Without this the enthalpy, which is tens of watt years per square
    metre, would dominate the arclength norm and the continuation would step
    across the fold instead of turning around it.
    """

    def __init__(self, params: SeaIceParams, parameter: str = "forcing",
                 observable: str = "thickness_min",
                 steps_per_year: int = 200, store_every: int = 2):
        if observable not in OBSERVABLES:
            raise ValueError(
                f"observable must be one of {OBSERVABLES}, got {observable!r}")
        self.base = params
        self.parameter = parameter
        self._observable = observable
        self.steps_per_year = steps_per_year
        self.store_every = store_every
        self.ndim = 2
        self.state_scale = np.array([LATENT_ICE, 1.0])

    # -- the map -------------------------------------------------------------- #

    def _model(self, p: np.ndarray) -> SeaIceColumn:
        return SeaIceColumn(self.base.with_control(
            **{self.parameter: np.asarray(p, float)}))

    def advance(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Advance a batch of states by one year."""
        return self._model(p).annual_map(np.atleast_2d(x),
                                         steps_per_year=self.steps_per_year)

    # -- diagnostics ---------------------------------------------------------- #

    def cycle(self, x: np.ndarray, p: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, SeaIceColumn]:
        """One year of the trajectory for a batch, with the model used."""
        model = self._model(p)
        t, trajectory = model.annual_cycle(
            np.atleast_2d(x), steps_per_year=self.steps_per_year,
            store_every=self.store_every)
        return t, trajectory, model

    def observable(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The chosen diagnostic of the annual cycle, one value per point."""
        _, trajectory, model = self.cycle(x, p)
        thickness = model.thickness(trajectory[..., 0])

        if self._observable == "thickness_min":
            return thickness.min(axis=0)
        if self._observable == "thickness_max":
            return thickness.max(axis=0)
        if self._observable == "thickness_mean":
            return thickness.mean(axis=0)
        if self._observable == "open_fraction":
            return model.open_fraction(trajectory[..., 0]).mean(axis=0)
        return trajectory[..., 1].mean(axis=0)

    def summary(self, x: np.ndarray, p: np.ndarray) -> dict:
        """Every diagnostic at once, for a batch of periodic states."""
        _, trajectory, model = self.cycle(x, p)
        enthalpy = trajectory[..., 0]
        thickness = model.thickness(enthalpy)
        return {
            "thickness_min": thickness.min(axis=0),
            "thickness_max": thickness.max(axis=0),
            "thickness_mean": thickness.mean(axis=0),
            "open_fraction": model.open_fraction(enthalpy).mean(axis=0),
            "deep_temperature": trajectory[..., 1].mean(axis=0),
            "ice_free_fraction": (thickness < 0.05).mean(axis=0),
        }
