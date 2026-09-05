"""Stage 7: where the ice goes, in area and in mass.

The column model says whether a threshold exists. It cannot say which parts of
the Arctic lose their ice first, because it has no geography. This stage runs the
spatially resolved model across a sequence of greenhouse forcings and records the
whole field at each, so that retreat can be followed in area and in mass at the
same time and located on the map.

The two measures are not interchangeable and that is the point of computing both.
Area responds where the cover is already thin, around the margins, and saturates
once those margins are gone. Mass responds wherever the ice thins at all, which
includes the thick interior that contributes almost nothing to area until very
late. A retreat that looks modest in the satellite record of extent can therefore
correspond to a much larger loss of ice.

The last region to keep its summer ice is recorded as well. It is a real feature
of the Arctic, the refuge north of Greenland and the Canadian Arctic Archipelago
where the drift piles ice against the coast, and following it across the
sequence shows whether it survives the threshold or disappears with everything
else.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import FORCING_PRESENT, RUNS_DIR, ensure_dirs
from arcticgnn.physics.seaice import SeaIceParams
from arcticgnn.spatial.grid import build_graph, build_grid
from arcticgnn.spatial.model import SpatialArcticModel

#: Greenhouse forcings at which the field is recorded, W m^-2 relative to the
#: current climate period.  The first is preindustrial.
FORCINGS = np.array([-FORCING_PRESENT, 0.0, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0])


def load(name: str) -> dict:
    path = RUNS_DIR / name
    if not path.exists():
        raise SystemExit(f"run the earlier stages first, {name} is missing")
    return json.loads(path.read_text())


def run_forcing(model: SpatialArcticModel, forcing: float,
                spin_up_years: float = 120.0, steps_per_year: int = 192
                ) -> dict:
    """Settle the field at one forcing and summarise its annual cycle."""
    warmed = model.with_forcing(float(forcing))
    _, settled = warmed.integrate(warmed.initial_state(),
                                  years=spin_up_years,
                                  steps_per_year=steps_per_year)
    times, cycle = warmed.integrate(settled[-1], years=1.0,
                                    steps_per_year=steps_per_year,
                                    store_every=steps_per_year // 12)

    enthalpy = cycle[:, :, 0]
    thickness = warmed.thickness(enthalpy)
    extent = warmed.extent_km2(enthalpy) / 1e6
    volume = warmed.volume_km3(enthalpy) / 1e3

    september = int(np.argmin(extent))
    march = int(np.argmax(extent))

    return {"forcing": float(forcing),
            "extent_max": float(extent.max()),
            "extent_min": float(extent.min()),
            "extent_mean": float(extent.mean()),
            "volume_max": float(volume.max()),
            "volume_min": float(volume.min()),
            "volume_mean": float(volume.mean()),
            "thickness_september": thickness[september].tolist(),
            "thickness_march": thickness[march].tolist(),
            "concentration_september":
                warmed.concentration(enthalpy[september]).tolist(),
            "month_september": int(round((times[september] % 1.0) * 12)) % 12 + 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spin-up", type=float, default=120.0)
    arguments = parser.parse_args()

    ensure_dirs()
    calibration = load("calibration.json")
    if "spatial" not in calibration:
        raise SystemExit("the spatial calibration is missing; rerun stage 1")

    spatial = calibration["spatial"]
    grid = build_grid(spatial.get("spacing_km", 100.0))
    graph = build_graph(grid)
    column = SeaIceParams().replace(
        longwave_offset=float(spatial["longwave_offset"]),
        longwave_annual=float(spatial["longwave_annual"]),
        albedo_scale=float(spatial.get("albedo_scale", 0.5)))
    model = SpatialArcticModel(grid, graph, column=column)

    print(f"{grid.n_cells} ocean cells, "
          f"{len(FORCINGS)} forcings from {FORCINGS.min():.1f} to "
          f"{FORCINGS.max():.1f} W m-2")

    states = []
    for forcing in FORCINGS:
        entry = run_forcing(model, float(forcing), spin_up_years=arguments.spin_up)
        states.append(entry)
        print(f"  {forcing:+5.2f} W m-2: extent "
              f"{entry['extent_max']:5.2f} to {entry['extent_min']:5.2f} "
              f"million km2, volume "
              f"{entry['volume_max']:5.1f} to {entry['volume_min']:5.1f} "
              f"thousand km3")

    # Where the summer ice survives longest, measured as the fraction of the
    # sequence in which each cell still carries ice at the seasonal minimum.
    survival = np.mean([np.array(s["thickness_september"]) > 0.05
                        for s in states], axis=0)

    output = {"forcings": FORCINGS.tolist(),
              "states": states,
              "survival": survival.tolist(),
              "forcing_present": FORCING_PRESENT,
              "n_cells": int(grid.n_cells),
              "spacing_km": float(grid.spacing_km),
              "latitude": grid.latitude.tolist()}

    path = RUNS_DIR / "retreat.json"
    path.write_text(json.dumps(output))
    print(f"written {path}")


if __name__ == "__main__":
    main()
