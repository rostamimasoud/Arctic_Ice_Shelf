"""Observational records used to calibrate and to test the models.

Two public products are used.  The satellite record of Northern Hemisphere sea
ice extent supplies the target that the spatial model is calibrated against and
the interannual variability that the trained emulators are tested on.  The
recorded greenhouse gas forcing supplies the control parameter that the
historical integrations are driven with, so the modelled and observed periods
refer to the same forcing.

Both are small text files served over plain HTTPS, and both are cached under the
project data directory after the first download.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DATA_DIR

#: Monthly Northern Hemisphere sea ice extent from the National Snow and Ice
#: Data Center Sea Ice Index, version 4.
NSIDC_MONTHLY_URL = ("https://noaadata.apps.nsidc.org/NOAA/G02135/north/"
                     "monthly/data/N_{month:02d}_extent_v4.0.csv")

#: Reference period over which the calibration targets are averaged.  It is the
#: standard climatological normal and lies wholly inside the satellite era.
REFERENCE_PERIOD = (1991, 2020)


@dataclass
class SeaIceRecord:
    """Monthly sea ice extent from the satellite record."""

    year: np.ndarray             # (n_years,)
    extent: np.ndarray           # (n_years, 12), million km^2, NaN where missing
    source: str

    def month(self, month: int) -> tuple[np.ndarray, np.ndarray]:
        """Year and extent for one calendar month, missing values dropped."""
        values = self.extent[:, month - 1]
        good = np.isfinite(values)
        return self.year[good], values[good]

    def climatology(self, period: tuple = REFERENCE_PERIOD) -> np.ndarray:
        """Mean seasonal cycle over ``period``, million km^2."""
        inside = (self.year >= period[0]) & (self.year <= period[1])
        with np.errstate(invalid="ignore"):
            return np.nanmean(self.extent[inside], axis=0)

    def trend(self, month: int, period: tuple = REFERENCE_PERIOD
              ) -> tuple[float, float]:
        """Least-squares trend and its standard error, million km^2 per decade."""
        years, values = self.month(month)
        inside = (years >= period[0]) & (years <= period[1])
        years, values = years[inside], values[inside]
        if years.size < 3:
            return float("nan"), float("nan")

        design = np.stack([np.ones_like(years, float), years.astype(float)], axis=1)
        coefficients, residuals, *_ = np.linalg.lstsq(design, values, rcond=None)
        prediction = design @ coefficients
        dof = years.size - 2
        variance = float(((values - prediction) ** 2).sum() / dof)
        covariance = variance * np.linalg.inv(design.T @ design)
        return float(coefficients[1] * 10.0), float(np.sqrt(covariance[1, 1]) * 10.0)


def _download(url: str, path: Path) -> Path:
    """Fetch ``url`` to ``path`` unless it is already there."""
    if path.exists() and path.stat().st_size > 0:
        return path
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=90) as response:
        payload = response.read()
    path.write_bytes(payload)
    return path


def load_sea_ice_extent(cache_dir: Path | None = None) -> SeaIceRecord:
    """Monthly Northern Hemisphere sea ice extent, million km^2.

    The Sea Ice Index is published one file per calendar month.  The files are
    fetched separately and assembled into a year by month table, with missing
    values left as not-a-number: the record has a gap in late 1987 and early
    1988 when the sensor failed, and filling it would invent data.
    """
    directory = Path(cache_dir) if cache_dir is not None else DATA_DIR / "nsidc"

    columns: dict[int, dict[int, float]] = {}
    for month in range(1, 13):
        path = _download(NSIDC_MONTHLY_URL.format(month=month),
                         directory / f"N_{month:02d}_extent_v4.0.csv")
        text = path.read_text().strip().splitlines()
        for line in text[1:]:
            fields = [f.strip() for f in line.split(",")]
            if len(fields) < 5:
                continue
            try:
                year = int(fields[0])
                extent = float(fields[4])
            except ValueError:
                continue
            # The file marks absent months with a negative sentinel.
            columns.setdefault(year, {})[month] = extent if extent > 0 else np.nan

    years = np.array(sorted(columns), int)
    table = np.full((years.size, 12), np.nan)
    for row, year in enumerate(years):
        for month, value in columns[year].items():
            table[row, month - 1] = value

    return SeaIceRecord(year=years, extent=table,
                        source="NSIDC Sea Ice Index v4, monthly extent")


# --------------------------------------------------------------------------- #
# Observed position of the ice edge
# --------------------------------------------------------------------------- #

#: Median position of the sea ice edge for each calendar month, as published
#: alongside the Sea Ice Index.  The edge is the 15 per cent concentration
#: contour, the same threshold used to define extent, so the line can be laid
#: directly over a modelled field without any further conversion.
NSIDC_MEDIAN_EDGE_URL = ("https://noaadata.apps.nsidc.org/NOAA/G02135/north/"
                         "monthly/shapefiles/shp_median/"
                         "median_extent_N_{month:02d}_1981-2010_polyline_v4.0.zip")

#: Reference period of the published median edge.  It differs from the period
#: used for the extent targets, and saying so matters: the edge is a 1981 to
#: 2010 median while the extent climatology is 1991 to 2020, so the edge sits
#: slightly further south than the later climatology alone would imply.
EDGE_PERIOD = (1981, 2010)


def load_median_ice_edge(month: int, cache_dir: Path | None = None) -> list:
    """Median position of the ice edge for one calendar month.

    Returns a list of arrays of shape ``(n, 2)`` holding latitude and longitude
    along each segment of the edge.  The published files are in the polar
    stereographic projection of the passive microwave grid, so they are
    reprojected to geographic coordinates here; that keeps the reprojection in
    one place and lets the plotting code work in the projection of the model
    grid.
    """
    directory = Path(cache_dir) if cache_dir is not None else DATA_DIR / "nsidc"
    archive = _download(NSIDC_MEDIAN_EDGE_URL.format(month=month),
                        directory / f"median_extent_N_{month:02d}.zip")

    import zipfile

    extracted = directory / f"median_extent_N_{month:02d}"
    if not extracted.exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)

    shapefiles = sorted(extracted.glob("*.shp"))
    if not shapefiles:
        raise RuntimeError(f"no shapefile inside {archive}")

    import shapefile as pyshp                     # from the pyshp package
    from pyproj import Transformer

    # The Sea Ice Index polar stereographic grid: true scale at 70 degrees
    # north, central meridian 45 degrees west, Hughes 1980 ellipsoid.
    transformer = Transformer.from_crs(
        "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 "
        "+a=6378273 +b=6356889.449 +units=m +no_defs",
        "EPSG:4326", always_xy=True)

    segments = []
    with pyshp.Reader(str(shapefiles[0])) as reader:
        for shape in reader.shapes():
            points = np.asarray(shape.points, float)
            if points.size == 0:
                continue
            parts = list(shape.parts) + [len(points)]
            for start, end in zip(parts[:-1], parts[1:]):
                piece = points[start:end]
                if piece.shape[0] < 2:
                    continue
                longitude, latitude = transformer.transform(piece[:, 0],
                                                            piece[:, 1])
                segments.append(np.stack([latitude, longitude], axis=1))
    return segments


# --------------------------------------------------------------------------- #
# Calibration targets
# --------------------------------------------------------------------------- #

@dataclass
class ExtentTargets:
    """Observed seasonal extremes of sea ice extent, million km^2."""

    maximum: float
    minimum: float
    month_maximum: int
    month_minimum: int
    period: tuple

    def describe(self) -> str:
        return (f"maximum {self.maximum:.2f} million km2 in month "
                f"{self.month_maximum}, minimum {self.minimum:.2f} in month "
                f"{self.month_minimum}, averaged over "
                f"{self.period[0]} to {self.period[1]}")


def extent_targets(record: SeaIceRecord,
                   period: tuple = REFERENCE_PERIOD) -> ExtentTargets:
    """Seasonal maximum and minimum extent averaged over the reference period."""
    seasonal = record.climatology(period)
    return ExtentTargets(maximum=float(np.nanmax(seasonal)),
                         minimum=float(np.nanmin(seasonal)),
                         month_maximum=int(np.nanargmax(seasonal)) + 1,
                         month_minimum=int(np.nanargmin(seasonal)) + 1,
                         period=period)
