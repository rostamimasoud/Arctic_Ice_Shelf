"""Solar radiation at the top of the atmosphere and at the surface.

Insolation is computed from orbital geometry instead of being prescribed as a
fitted cycle, because the spatial model needs the latitude dependence to be
right: the contrast between the pole, which is dark for months, and the sub
Arctic seas, which are lit for most of the year, is what sets where ice survives
the summer.  A fitted cycle applied uniformly would remove that contrast and
with it the spatial structure the emulators are meant to learn.
"""

from __future__ import annotations

import numpy as np

#: Solar constant, W m^-2.
SOLAR_CONSTANT = 1361.0

#: Obliquity of the ecliptic, degrees.
OBLIQUITY = 23.44

#: Day of the year of the northern summer solstice.
SOLSTICE_DAY = 172.0

#: Bulk shortwave transmission of the Arctic atmosphere, including cloud.  The
#: Arctic is heavily overcast in summer, so this is well below a clear-sky
#: value; it is the single number that scales surface shortwave from the value
#: at the top of the atmosphere.
TRANSMISSION = 0.52


def declination(t_years: np.ndarray) -> np.ndarray:
    """Solar declination in radians, with time in years from 1 January."""
    day = np.asarray(t_years, float) * 365.25
    angle = 2.0 * np.pi * (day - SOLSTICE_DAY) / 365.25
    return np.radians(OBLIQUITY) * np.cos(angle)


def daily_insolation(latitude: np.ndarray, t_years: np.ndarray) -> np.ndarray:
    """Daily mean insolation at the top of the atmosphere, W m^-2.

    Returns an array of shape ``(len(t_years), len(latitude))``.  Polar night
    and polar day are handled by the limits of the sunrise hour angle, so the
    long dark winter appears without any special casing.
    """
    latitude = np.radians(np.asarray(latitude, float))[None, :]
    delta = declination(t_years)[:, None]

    # Cosine of the sunrise hour angle, clipped to give continuous polar day and
    # polar night at the two limits.
    cosine = -np.tan(latitude) * np.tan(delta)
    hour_angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return (SOLAR_CONSTANT / np.pi) * (
        hour_angle * np.sin(latitude) * np.sin(delta)
        + np.cos(latitude) * np.cos(delta) * np.sin(hour_angle))


def surface_shortwave(latitude: np.ndarray, t_years: np.ndarray,
                      transmission: float = TRANSMISSION) -> np.ndarray:
    """Downward shortwave flux at the surface, W m^-2."""
    return transmission * daily_insolation(latitude, t_years)
