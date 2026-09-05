"""Stage 2: the bifurcation structure of the calibrated column model.

Three analyses run here, in order.

First, for every calibrated configuration the branch of annual cycles is traced
as the greenhouse forcing is varied.  Continuation follows the branch around its
turning points, so the unstable states between the perennial and the seasonally
ice free branch are recovered as well, and the two are shown to be parts of one
S shaped curve rather than separate solutions.

Second, the saddle-node bifurcations on each branch are refined and the width of
the bistable window between them is measured.  This gives the forcing at which
the ice is lost, the forcing to which it would have to be returned to recover
it, and the size of the jump.

Third, the regime is mapped over a plane of two control parameters, the
greenhouse forcing and the heat supplied by inflowing Atlantic Water.  This is
the overall picture: it shows which combinations support perennial ice, which
support only a seasonal cover, and where the two coexist.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import (FORCING_PRESENT, FORCING_PRESENT_RANGE,
                              RUNS_DIR, ensure_dirs, forcing_to_warming)
from arcticgnn.dynsys.continuation import bifurcation_diagram
from arcticgnn.dynsys.problems import SeaIceProblem
from arcticgnn.physics.seaice import SeaIceColumn, SeaIceParams

#: Range of greenhouse forcing explored, W m^-2.
FORCING_RANGE = (-6.0, 16.0)

#: Atlantic Water heat supply examined in the regime map, expressed as the
#: inflow temperature above freezing, K.
ATLANTIC_RANGE = (1.2, 4.4)


def load_calibration() -> dict:
    path = RUNS_DIR / "calibration.json"
    if not path.exists():
        raise SystemExit("run the calibration stage first")
    return json.loads(path.read_text())


def configuration(calibration: dict, index: int) -> SeaIceParams:
    """Parameters of one calibrated configuration."""
    column = calibration["column"]
    return SeaIceParams().replace(
        albedo_scale=float(column["albedo_scale"][index]),
        longwave_offset=float(column["longwave_offset"][index]),
        longwave_annual=float(column["longwave_annual"][index]),
        forcing=0.0)


def trace_configuration(params: SeaIceParams, parameter: str = "forcing",
                        start_value: float = 0.0,
                        limits: tuple = FORCING_RANGE,
                        max_steps: int = 1200,
                        verbose: bool = True) -> dict:
    """Continue the branch of annual cycles in one control parameter."""
    problem = SeaIceProblem(params, parameter=parameter,
                            observable="thickness_min")

    # The continuation starts from the calibrated state of today, which lies on
    # the perennial branch.
    model = SeaIceColumn(params.with_control(**{parameter: start_value}))
    start = model.spin_up(model.initial_state("perennial"), years=300.0,
                          steps_per_year=problem.steps_per_year)

    branch, folds, window = bifurcation_diagram(
        problem, start, start_value, limits[0], limits[1], ds=0.03,
        max_steps=max_steps)

    summary = problem.summary(branch.x, branch.p)
    result = {
        "albedo_scale": float(params.albedo_scale),
        "parameter": parameter,
        "forcing": branch.p.tolist(),
        "thickness_min": branch.observable.tolist(),
        "thickness_max": summary["thickness_max"].tolist(),
        "thickness_mean": summary["thickness_mean"].tolist(),
        "deep_temperature": summary["deep_temperature"].tolist(),
        "ice_free_fraction": summary["ice_free_fraction"].tolist(),
        "stable": branch.stable.tolist(),
        "multiplier": branch.multiplier.tolist(),
        "folds": [{"forcing": f.p, "thickness_min": f.observable,
                   "direction": f.direction, "refined": bool(f.refined),
                   "residual": f.residual} for f in folds],
    }
    if window is not None:
        result["hysteresis"] = {
            "forcing_forward": window.p_forward,
            "forcing_reverse": window.p_reverse,
            "width": window.width,
            "width_kelvin": forcing_to_warming(window.width),
            "jump": window.jump}
    if verbose:
        n_unstable = int((~branch.stable).sum())
        print(f"  {parameter} at scale {params.albedo_scale:.2f} m: "
              f"{len(branch)} points, "
              f"{n_unstable} unstable, {len(folds)} folds"
              + (f", hysteresis {window.width:.2f} W m-2"
                 if window is not None else ", no hysteresis"))
    return result


def coexisting_cycles(calibration: dict, index: int, branch: dict,
                      steps_per_year: int = 200) -> dict:
    """The two annual cycles that coexist at one forcing inside the window.

    Both are obtained at the same forcing, one by relaxing from thick perennial
    ice and one from an ice free ocean. Showing them side by side is what makes
    bistability concrete: two different years, indefinitely repeatable, under
    identical conditions.
    """
    window = branch.get("hysteresis")
    if window is None:
        return {}

    # The middle of the window, where the two states are furthest apart and
    # neither sits uncomfortably close to a fold.
    forcing = 0.5 * (window["forcing_forward"] + window["forcing_reverse"])
    params = configuration(calibration, index).replace(forcing=forcing)
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
            "deep_temperature": trajectory[:, 1].tolist(),
            "open_fraction": model.open_fraction(trajectory[:, 0]).tolist()}

    return {"forcing": float(forcing),
            "albedo_scale": float(params.albedo_scale),
            "cycles": cycles}


def regime_map(params: SeaIceParams, n_forcing: int = 34,
               n_atlantic: int = 26, spin_up_years: float = 250.0,
               steps_per_year: int = 200, verbose: bool = True) -> dict:
    """Map which regimes are attractors over the two control parameters.

    Every point of the plane is relaxed twice, once from thick perennial ice and
    once from an ice free ocean.  Where the two settle on different cycles the
    point is bistable.  Running the whole plane as one batched integration makes
    this affordable.
    """
    forcings = np.linspace(FORCING_RANGE[0], min(FORCING_RANGE[1], 12.0),
                           n_forcing)
    atlantic = np.linspace(ATLANTIC_RANGE[0], ATLANTIC_RANGE[1], n_atlantic)
    forcing_grid, atlantic_grid = np.meshgrid(forcings, atlantic, indexing="ij")
    flat_forcing = forcing_grid.ravel()
    flat_atlantic = atlantic_grid.ravel()

    model = SeaIceColumn(params.replace(forcing=flat_forcing,
                                        atlantic_temperature=flat_atlantic))

    outcome = {}
    for regime in ("perennial", "seasonal"):
        single = model.initial_state(regime)
        start = np.repeat(single[None, :], flat_forcing.size, axis=0)
        settled = model.spin_up(start, years=spin_up_years,
                                steps_per_year=steps_per_year)
        _, cycle = model.annual_cycle(settled, steps_per_year=steps_per_year,
                                      store_every=2)
        thickness = model.thickness(cycle[..., 0])
        outcome[regime] = {
            "thickness_min": thickness.min(axis=0).reshape(forcing_grid.shape),
            "thickness_max": thickness.max(axis=0).reshape(forcing_grid.shape)}
        if verbose:
            print(f"  relaxed the plane from the {regime} state")

    perennial = outcome["perennial"]["thickness_min"] > 0.05
    seasonal = outcome["seasonal"]["thickness_min"] <= 0.05
    # A point is bistable when the two starting states settle on different
    # cycles, one keeping ice through the summer and one not.
    bistable = perennial & seasonal

    return {"forcing": forcings.tolist(),
            "atlantic_temperature": atlantic.tolist(),
            "thickness_min_from_perennial":
                outcome["perennial"]["thickness_min"].tolist(),
            "thickness_min_from_seasonal":
                outcome["seasonal"]["thickness_min"].tolist(),
            "thickness_max_from_perennial":
                outcome["perennial"]["thickness_max"].tolist(),
            "perennial_possible": perennial.tolist(),
            "bistable": bistable.tolist(),
            "albedo_scale": float(params.albedo_scale)}


def parameter_plane(calibration: dict, indices: list, n_forcing: int = 40,
                    spin_up_years: float = 250.0, steps_per_year: int = 200,
                    verbose: bool = True) -> dict:
    """Map the regimes over greenhouse forcing and the albedo transition scale.

    This is the plane the result lives in.  Each row uses the surface energy
    budget that was calibrated for that transition scale, so every row
    reproduces the same observed present-day Arctic and the rows differ only in
    the parameter under study.  Comparing rows therefore isolates its effect,
    which a plane of two parameters swept without recalibration could not do.

    Every point is relaxed twice, from thick perennial ice and from an ice free
    ocean, and the whole plane is integrated as one batch.
    """
    column = calibration["column"]
    scales = np.array([column["albedo_scale"][i] for i in indices])
    offsets = np.array([column["longwave_offset"][i] for i in indices])
    forcings = np.linspace(-6.0, 6.0, n_forcing)

    forcing_grid, index_grid = np.meshgrid(forcings, np.arange(scales.size),
                                           indexing="ij")
    flat_forcing = forcing_grid.ravel()
    flat_scale = scales[index_grid.ravel()]
    flat_offset = offsets[index_grid.ravel()]

    model = SeaIceColumn(SeaIceParams().replace(
        albedo_scale=flat_scale, longwave_offset=flat_offset,
        forcing=flat_forcing))

    outcome = {}
    for regime in ("perennial", "seasonal"):
        single = model.initial_state(regime)
        start = np.repeat(single[None, :], flat_forcing.size, axis=0)
        settled = model.spin_up(start, years=spin_up_years,
                                steps_per_year=steps_per_year)
        _, cycle = model.annual_cycle(settled, steps_per_year=steps_per_year,
                                      store_every=2)
        thickness = model.thickness(cycle[..., 0])
        outcome[regime] = thickness.min(axis=0).reshape(forcing_grid.shape)
        if verbose:
            print(f"  relaxed the plane from the {regime} state")

    perennial = outcome["perennial"] > 0.05
    seasonal = outcome["seasonal"] <= 0.05
    bistable = perennial & seasonal

    return {"forcing": forcings.tolist(),
            "albedo_scale": scales.tolist(),
            "thickness_min_from_perennial": outcome["perennial"].tolist(),
            "thickness_min_from_seasonal": outcome["seasonal"].tolist(),
            "perennial_possible": perennial.tolist(),
            "bistable": bistable.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-branches", action="store_true")
    parser.add_argument("--skip-map", action="store_true")
    arguments = parser.parse_args()

    ensure_dirs()
    calibration = load_calibration()
    feasible = [i for i, ok in enumerate(calibration["column"]["converged"]) if ok]
    print(f"{len(feasible)} calibrated configurations")

    output = {"albedo_scale": [calibration["column"]["albedo_scale"][i]
                               for i in feasible]}

    if not arguments.skip_branches:
        print("continuing the branch of annual cycles in greenhouse forcing")
        output["branches"] = [trace_configuration(configuration(calibration, i))
                              for i in feasible]

    # A parameter that carries no fold, for comparison.  Showing one of each
    # makes clear that bistability is a property of particular controls and not
    # of the model as a whole.
    widest = max(range(len(output.get("branches", []))),
                 key=lambda i: (output["branches"][i].get("hysteresis")
                                or {"width": -1.0})["width"],
                 default=len(feasible) // 2)
    chosen = feasible[widest]
    params = configuration(calibration, chosen)

    if not arguments.skip_branches:
        print("continuing the same configuration in the Atlantic inflow "
              "temperature, which carries no fold")
        output["monotone_branch"] = trace_configuration(
            params, parameter="atlantic_temperature",
            start_value=params.atlantic_temperature, limits=(0.5, 5.0),
            max_steps=700)

        print("the two annual cycles that coexist inside the window")
        output["coexisting"] = coexisting_cycles(calibration, chosen,
                                                 output["branches"][widest])

    if not arguments.skip_map:
        print("mapping the regimes over forcing and the albedo transition scale")
        output["parameter_plane"] = parameter_plane(calibration, feasible)

        # The plane of forcing against ocean heat supply is drawn for the
        # configuration with the widest bistable window, since that is where a
        # change in ocean heat can be seen to move the threshold.
        print(f"mapping forcing against ocean heat supply for a transition "
              f"scale of {calibration['column']['albedo_scale'][chosen]:.2f} m")
        output["regime_map"] = regime_map(configuration(calibration, chosen))

    # How much of the way from preindustrial conditions to each threshold the
    # current climate period has already covered.
    if output.get("branches"):
        print("distance from preindustrial conditions to each threshold")
        travelled = []
        for scale, branch in zip(output["albedo_scale"], output["branches"]):
            window = branch.get("hysteresis")
            if window is None:
                travelled.append(None)
                continue
            threshold = float(window["forcing_forward"])
            span = threshold + FORCING_PRESENT
            entry = {
                "albedo_scale": scale,
                "threshold_from_present": threshold,
                "threshold_from_preindustrial": span,
                "fraction_travelled": FORCING_PRESENT / span if span > 0 else None,
                "fraction_range": [low / (threshold + low)
                                   for low in FORCING_PRESENT_RANGE]}
            travelled.append(entry)
            print(f"  scale {scale:.2f} m: threshold {span:.2f} W m-2 above "
                  f"preindustrial, of which "
                  f"{100 * entry['fraction_travelled']:.0f} per cent has been "
                  f"covered")
        output["distance_travelled"] = travelled
        output["forcing_present"] = FORCING_PRESENT
        output["forcing_present_range"] = list(FORCING_PRESENT_RANGE)

    path = RUNS_DIR / "bifurcation.json"
    path.write_text(json.dumps(output))
    print(f"written {path}")


if __name__ == "__main__":
    main()
