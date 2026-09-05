"""Stage 6: the basins of attraction, and the two Arctics they describe.

The branch diagram says how many states exist at a given forcing. It does not
say how far the system sits from the boundary between them, and that distance is
what decides whether a disturbance of a given size matters. This stage maps the
boundary directly.

The state of the column is two numbers, the surface enthalpy and the temperature
of the water below the halocline, so the set of possible states is a plane and
can be drawn. Every point of a grid over that plane is taken as an initial
condition, integrated until it settles, and labelled by which annual cycle it
reached. Where the labels change is the boundary between the basins, and it is
the unstable cycle recovered by the continuation that separates them.

The same computation is carried out for two configurations that reproduce the
Arctic as it is observed today equally well, one holding a single state and one
holding two. Placing them beside each other shows what the study is about: two
systems that agree on the present and disagree entirely about what a forcing
would do to them.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import LATENT_ICE, RUNS_DIR, ensure_dirs
from arcticgnn.physics.seaice import ICE_FREE_THRESHOLD, SeaIceColumn, SeaIceParams


def load(name: str) -> dict:
    path = RUNS_DIR / name
    if not path.exists():
        raise SystemExit(f"run the earlier stages first, {name} is missing")
    return json.loads(path.read_text())


def configuration(calibration: dict, index: int,
                  forcing: float = 0.0) -> SeaIceParams:
    column = calibration["column"]
    return SeaIceParams().replace(
        albedo_scale=float(column["albedo_scale"][index]),
        longwave_offset=float(column["longwave_offset"][index]),
        longwave_annual=float(column["longwave_annual"][index]),
        forcing=float(forcing))


def basin_map(params: SeaIceParams, thickness_limit: float = 4.0,
              deep_limit: tuple = (0.0, 3.6), n_thickness: int = 56,
              n_deep: int = 56, years: float = 300.0,
              steps_per_year: int = 200) -> dict:
    """Label each starting state by the annual cycle it settles onto.

    The whole plane is integrated as one batch, so a grid of a few thousand
    starting states costs one long integration instead of a few thousand short
    ones.
    """
    thickness = np.linspace(0.0, thickness_limit, n_thickness)
    deep = np.linspace(deep_limit[0], deep_limit[1], n_deep)
    thickness_grid, deep_grid = np.meshgrid(thickness, deep, indexing="ij")

    start = np.stack([-thickness_grid.ravel() * LATENT_ICE,
                      deep_grid.ravel()], axis=-1)

    model = SeaIceColumn(params)
    settled = model.spin_up(start, years=years, steps_per_year=steps_per_year)
    _, cycle = model.annual_cycle(settled, steps_per_year=steps_per_year,
                                  store_every=4)
    minimum = model.thickness(cycle[..., 0]).min(axis=0)

    perennial = (minimum > ICE_FREE_THRESHOLD).reshape(thickness_grid.shape)
    return {"thickness": thickness.tolist(),
            "deep_temperature": deep.tolist(),
            "perennial": perennial.tolist(),
            "thickness_min": minimum.reshape(thickness_grid.shape).tolist(),
            "forcing": float(np.atleast_1d(params.forcing)[0]),
            "albedo_scale": float(params.albedo_scale),
            "fraction_perennial": float(perennial.mean())}


def limit_cycles(params: SeaIceParams, steps_per_year: int = 200) -> dict:
    """The annual cycles the system settles onto, one per regime it holds."""
    model = SeaIceColumn(params)
    cycles = {}
    for regime in ("perennial", "seasonal"):
        settled = model.spin_up(model.initial_state(regime), years=400.0,
                                steps_per_year=steps_per_year)
        times, trajectory = model.annual_cycle(settled,
                                               steps_per_year=steps_per_year,
                                               store_every=2)
        cycles[regime] = {
            "time": times.tolist(),
            "thickness": model.thickness(trajectory[:, 0]).tolist(),
            "deep_temperature": trajectory[:, 1].tolist()}
    return cycles


def seasonal_cycle(params: SeaIceParams, steps_per_year: int = 200) -> dict:
    """The annual cycle of the perennial state, for comparison with observations."""
    model = SeaIceColumn(params)
    settled = model.spin_up(model.initial_state("perennial"), years=400.0,
                            steps_per_year=steps_per_year)
    times, trajectory = model.annual_cycle(settled,
                                           steps_per_year=steps_per_year,
                                           store_every=2)
    return {"time": times.tolist(),
            "thickness": model.thickness(trajectory[:, 0]).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=56)
    arguments = parser.parse_args()

    ensure_dirs()
    calibration = load("calibration.json")
    bifurcation = load("bifurcation.json")

    feasible = [i for i, ok in enumerate(calibration["column"]["converged"])
                if ok]
    branches = bifurcation["branches"]
    widths = [(branch.get("hysteresis") or {"width": 0.0})["width"]
              for branch in branches]

    # One configuration with the widest window and one with none at all.  Both
    # were calibrated to the same observed thickness.
    bistable = int(np.argmax(widths))
    monostable = int(np.argmin(widths))

    output = {"pair": []}
    for label, index in (("monostable", monostable), ("bistable", bistable)):
        window = branches[index].get("hysteresis")
        # The basins are mapped inside the window where one exists, and at the
        # same forcing for the other configuration so that the two panels are
        # directly comparable.
        forcing = (0.5 * (window["forcing_forward"] + window["forcing_reverse"])
                   if window else 0.0)
        entry = {"label": label,
                 "albedo_scale": calibration["column"]["albedo_scale"][
                     feasible[index]],
                 "window": window,
                 "forcing": forcing}
        params = configuration(calibration, feasible[index], forcing)
        print(f"{label}: albedo scale {entry['albedo_scale']:.2f} m, "
              f"mapping basins at {forcing:.2f} W m-2")
        entry["basins"] = basin_map(params, n_thickness=arguments.resolution,
                                    n_deep=arguments.resolution)
        entry["cycles"] = limit_cycles(params)
        entry["control_cycle"] = seasonal_cycle(
            configuration(calibration, feasible[index], 0.0))
        entry["branch_index"] = index
        print(f"  {entry['basins']['fraction_perennial']:.0%} of the plane "
              f"leads to perennial ice")
        output["pair"].append(entry)

    path = RUNS_DIR / "basins.json"
    path.write_text(json.dumps(output))
    print(f"written {path}")


if __name__ == "__main__":
    main()
