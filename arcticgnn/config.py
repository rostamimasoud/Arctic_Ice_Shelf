"""Paths, physical constants and unit conventions.

Energy bookkeeping follows the convention of column sea-ice models: surface
enthalpy is carried in W yr m^-2 rather than J m^-2, so that every flux in the
budget is already in W m^-2 and time is measured in years.  Conversions between
the two are collected in :data:`SECONDS_PER_YEAR`.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Project root.  Override with the ARCTICGNN_ROOT environment variable.
ROOT = Path(os.environ.get(
    "ARCTICGNN_ROOT", Path(__file__).resolve().parents[2])).expanduser()

DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
FIGURE_DIR = ROOT / "manuscript" / "figures"


def ensure_dirs() -> None:
    """Create the output directories if they do not yet exist."""
    for d in (DATA_DIR, RUNS_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Physical constants
# --------------------------------------------------------------------------- #

SECONDS_PER_YEAR = 3.15576e7        # s yr^-1 (365.25 d)

RHO_ICE = 917.0                     # kg m^-3
RHO_SEAWATER = 1025.0               # kg m^-3
LATENT_FUSION = 3.34e5              # J kg^-1
HEAT_CAPACITY_SEAWATER = 3985.0     # J kg^-1 K^-1
CONDUCTIVITY_ICE = 2.0              # W m^-1 K^-1

#: Volumetric latent heat of sea ice, W yr m^-3.  Ice thickness is -E / LATENT_ICE.
LATENT_ICE = RHO_ICE * LATENT_FUSION / SECONDS_PER_YEAR      # ~9.7

#: Radiative forcing of a CO2 doubling, W m^-2.
FORCING_2XCO2 = 3.71

#: Total anthropogenic effective radiative forcing of the current climate period,
#: 1991 to 2020, relative to 1750, in W m^-2.  Taken as the mean over that period
#: of the assessed history, which is smaller than the value for any single recent
#: year because the forcing grew throughout it.  The models here are calibrated
#: to observations of the current climate period, so this number is what places
#: their zero of forcing on an axis anchored at preindustrial conditions, and it
#: is what allows the distance already travelled towards a threshold to be
#: expressed as a fraction rather than only as a remaining margin.
FORCING_PRESENT = 2.30

#: Range of that estimate, W m^-2, carried through wherever it is used.
FORCING_PRESENT_RANGE = (1.90, 2.70)

#: Global climate feedback parameter, W m^-2 K^-1.  Used only to express
#: radiative forcing as an equivalent global-mean warming for the figures.
FEEDBACK_GLOBAL = 1.20


def heat_capacity(depth_m: float) -> float:
    """Heat capacity of a well-mixed water column, W yr m^-2 K^-1."""
    return (RHO_SEAWATER * HEAT_CAPACITY_SEAWATER * depth_m
            / SECONDS_PER_YEAR)


def forcing_to_warming(forcing: float) -> float:
    """Express a radiative forcing as an equivalent global-mean warming, K."""
    return forcing / FEEDBACK_GLOBAL


def warming_to_forcing(warming: float) -> float:
    """Inverse of :func:`forcing_to_warming`."""
    return warming * FEEDBACK_GLOBAL


def forcing_to_co2(forcing: float, co2_reference: float = 284.0) -> float:
    """Carbon dioxide concentration giving ``forcing``, parts per million."""
    import numpy as np
    return co2_reference * np.exp(forcing / FORCING_2XCO2 * np.log(2.0))


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

#: Base seed.  Every stochastic routine takes an explicit seed defaulting to a
#: fixed offset from this, so a whole-pipeline rerun is bit-reproducible.
SEED = 20260101
