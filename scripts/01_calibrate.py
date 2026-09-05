"""Stage 1: calibrate both models against observations.

Both models have exactly one free quantity in their surface energy budget, the
temperature-independent part of the net upward flux, and each is set by matching
one observed number.  The seasonal amplitude of that flux is fixed beforehand
from the observed non-solar surface fluxes over Arctic sea ice and is never
fitted.  That restraint is the whole point of the stage.  Given two free
parameters and two thickness targets the solver will happily find a budget in
which the winter surface gains heat, matching every observed number while
describing no Arctic at all.  With the amplitude pinned, whatever the model then
produces beyond its single target is a prediction that can be tested.

The stage runs in three parts.

First the column model is calibrated once for each albedo transition scale, to
the observed annual mean thickness of the central Arctic.  Holding every
configuration to the same present-day climate is what later allows differences
in tipping behaviour to be attributed to that parameter instead of to
differences in the control state.  The seasonal range of thickness that each
configuration produces is recorded for comparison with observations.

Second, the offset is swept widely at each scale and the closest approach to the
observations is recorded, which shows whether a scale can describe the
present-day Arctic at all.

Third the spatial model is calibrated to the observed September extent, leaving
the observed March extent as an independent check.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import RUNS_DIR, ensure_dirs
from arcticgnn.data.observations import extent_targets, load_sea_ice_extent
from arcticgnn.physics.calibrate import (TARGET_THICKNESS_MAX,
                                         TARGET_THICKNESS_MEAN,
                                         TARGET_THICKNESS_MIN,
                                         calibrate_offset, mean_thickness)
from arcticgnn.physics.seaice import SeaIceParams
from arcticgnn.spatial.grid import build_graph, build_grid
from arcticgnn.spatial.model import SpatialArcticModel

#: Albedo transition scales examined, m.
ALBEDO_SCALES = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70,
                          0.72, 0.74, 0.76, 0.78, 0.80])


def calibrate_column(verbose: bool = True) -> dict:
    """Calibrate the column model across the albedo transition scales.

    One quantity is adjusted, the temperature-independent part of the net upward
    surface flux, and one target is matched, the observed annual mean thickness.
    The seasonal amplitude of that flux is held at the value implied by the
    observed non-solar fluxes and is deliberately not fitted.  Leaving it free
    lets the solver reach the thickness targets by driving the amplitude towards
    zero, which closes the budget only by making the winter surface gain heat.
    That solution matches the observed thickness and describes no Arctic, so the
    seasonal range of thickness is reported here as a prediction to be checked
    against observations instead of as something the calibration was given.
    """
    results = [calibrate_offset(SeaIceParams(), float(scale),
                                spin_up_years=250.0, verbose=False)
               for scale in ALBEDO_SCALES]

    if verbose:
        print("column calibration, seasonal flux amplitude held at "
              f"{SeaIceParams().longwave_annual:.1f} W m-2")
        print("  scale   offset   mean    h_max   h_min   error   ok")
        for scale, r in zip(ALBEDO_SCALES, results):
            mean = 0.5 * (r.thickness_max + r.thickness_min)
            print(f"  {scale:5.2f}  {r.longwave_offset:7.2f} {mean:6.2f} "
                  f"{r.thickness_max:7.2f} {r.thickness_min:7.2f} "
                  f"{r.residual:7.3f}   {r.converged}")

    return {"albedo_scale": ALBEDO_SCALES.tolist(),
            "longwave_offset": [r.longwave_offset for r in results],
            "longwave_annual": [r.longwave_annual for r in results],
            "thickness_max": [r.thickness_max for r in results],
            "thickness_min": [r.thickness_min for r in results],
            "converged": [bool(r.converged) for r in results],
            "target_mean": TARGET_THICKNESS_MEAN,
            "target_max": TARGET_THICKNESS_MAX,
            "target_min": TARGET_THICKNESS_MIN}


def feasibility_scan(n_offset: int = 27, verbose: bool = True) -> dict:
    """Record how closely each albedo transition scale can match observations.

    The offset is swept over a wide range at each scale and the closest approach
    to the observed annual mean thickness is kept.  A scale whose best approach
    is still far from the observations cannot describe the present-day Arctic,
    which is a statement about the parameter and not about the solver.
    """
    offsets = np.linspace(52.0, 110.0, n_offset)
    best_error, best_offset, curves = [], [], []
    thinnest, gap = [], []

    for scale in ALBEDO_SCALES:
        params = SeaIceParams().replace(albedo_scale=float(scale))
        mean, high, low = mean_thickness(params, offsets, spin_up_years=300.0)
        error = np.where(mean > 0.05, np.abs(mean - TARGET_THICKNESS_MEAN),
                         np.inf)
        best = int(np.argmin(error))
        best_error.append(float(error[best]))
        best_offset.append(float(offsets[best]))
        curves.append({"mean": mean.tolist(), "max": high.tolist(),
                       "min": low.tolist()})

        # The thinnest perennial cover the branch supports, and the size of the
        # gap the fold opens beneath it.  Where that gap straddles the observed
        # thickness, no surface energy budget can reproduce today's Arctic at
        # this albedo transition scale, which rules the value out.
        surviving = mean[mean > 0.05]
        floor = float(surviving.min()) if surviving.size else float("nan")
        thinnest.append(floor)
        gap.append(float(max(floor - TARGET_THICKNESS_MEAN, 0.0)))
        if verbose:
            print(f"  scale {scale:.2f} m: thinnest perennial state "
                  f"{floor:.2f} m, closest approach to the observed "
                  f"{TARGET_THICKNESS_MEAN:.2f} m is {error[best]:.3f} m")

    return {"albedo_scale": ALBEDO_SCALES.tolist(),
            "best_error": best_error, "best_offset": best_offset,
            "thinnest_perennial": thinnest, "gap_above_observed": gap,
            "target_mean": TARGET_THICKNESS_MEAN,
            "offsets": offsets.tolist(), "curves": curves}


def _spatial_extremes(offset: float, annual: float, spacing_km: float,
                      spin_up_years: float, steps_per_year: int,
                      albedo_scale: float = 0.5) -> tuple[float, float]:
    """Seasonal maximum and minimum extent of the spatial model, million km^2."""
    grid = build_grid(spacing_km)
    graph = build_graph(grid)
    params = SeaIceParams().replace(longwave_offset=float(offset),
                                    longwave_annual=float(annual),
                                    albedo_scale=float(albedo_scale))
    model = SpatialArcticModel(grid, graph, column=params)
    _, settled = model.integrate(model.initial_state(), years=spin_up_years,
                                 steps_per_year=steps_per_year)
    _, cycle = model.integrate(settled[-1], years=1.0,
                               steps_per_year=steps_per_year, store_every=4)
    extent = model.extent_km2(cycle[:, :, 0]) / 1e6
    return float(extent.max()), float(extent.min())


def calibrate_spatial(targets, spacing_km: float = 100.0,
                      spin_up_years: float = 90.0, steps_per_year: int = 200,
                      bracket: tuple = (72.0, 88.0), n_coarse: int = 5,
                      tolerance: float = 0.15, max_iterations: int = 9,
                      albedo_scale: float = 0.5, verbose: bool = True) -> dict:
    """Calibrate the spatial model to the observed sea ice extent.

    As in the column model a single quantity is adjusted, the offset of the net
    upward surface flux, with its seasonal amplitude held at the observed value.
    The target is the observed seasonal minimum extent, reached in September,
    because that is the quantity the ice-albedo feedback controls and the one
    that carries the strongest observed trend.  The seasonal maximum is then a
    prediction, and comparing it against the satellite record tests the
    calibration rather than restating it.

    The search is a coarse sweep followed by bisection.  Extent rises
    monotonically with the offset, and bisection cannot be thrown off by the
    region where no summer ice survives and the response goes flat.
    """
    annual = SeaIceParams().longwave_annual

    def evaluate(offset: float) -> tuple:
        return _spatial_extremes(offset, annual, spacing_km, spin_up_years,
                                 steps_per_year, albedo_scale)

    if verbose:
        print(f"  coarse sweep, seasonal flux amplitude held at {annual:.1f}")
    grid = np.linspace(bracket[0], bracket[1], n_coarse)
    sweep = []
    for offset in grid:
        maximum, minimum = evaluate(float(offset))
        sweep.append({"offset": float(offset), "extent_max": maximum,
                      "extent_min": minimum})
        if verbose:
            print(f"    offset {offset:5.1f} -> maximum {maximum:5.2f} "
                  f"minimum {minimum:5.2f} million km2")

    minima = np.array([entry["extent_min"] for entry in sweep])
    below = np.where(minima <= targets.minimum)[0]
    above = np.where(minima >= targets.minimum)[0]
    if below.size == 0 or above.size == 0:
        best = int(np.argmin(np.abs(minima - targets.minimum)))
        chosen = sweep[best]
        return {"longwave_offset": chosen["offset"], "longwave_annual": annual,
                "extent_max": chosen["extent_max"],
                "extent_min": chosen["extent_min"],
                "target_max": targets.maximum, "target_min": targets.minimum,
                "converged": False, "sweep": sweep, "history": [],
                "spacing_km": spacing_km, "albedo_scale": albedo_scale,
                "spin_up_years": spin_up_years, "steps_per_year": steps_per_year}

    left = float(grid[below.max()])
    right = float(grid[above.min()])
    history = []
    for iteration in range(1, max_iterations + 1):
        middle = 0.5 * (left + right)
        maximum, minimum = evaluate(middle)
        error = minimum - targets.minimum
        history.append({"iteration": iteration, "offset": middle,
                        "extent_max": maximum, "extent_min": minimum,
                        "error": float(abs(error))})
        if verbose:
            print(f"  iteration {iteration:2d}: offset {middle:6.2f} -> "
                  f"maximum {maximum:5.2f} minimum {minimum:5.2f} "
                  f"(error {abs(error):.3f})")
        if abs(error) < tolerance:
            break
        if error < 0.0:
            left = middle
        else:
            right = middle

    final = history[-1]
    return {"longwave_offset": final["offset"], "longwave_annual": annual,
            "extent_max": final["extent_max"], "extent_min": final["extent_min"],
            "target_max": targets.maximum, "target_min": targets.minimum,
            "converged": bool(final["error"] < tolerance),
            "sweep": sweep, "history": history, "spacing_km": spacing_km,
            "albedo_scale": albedo_scale, "spin_up_years": spin_up_years,
            "steps_per_year": steps_per_year}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-column", action="store_true")
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-spatial", action="store_true")
    parser.add_argument("--spacing", type=float, default=100.0)
    arguments = parser.parse_args()

    ensure_dirs()
    # Merge into whatever is already stored rather than replacing it, so that
    # skipping a part of the stage leaves the results of that part intact
    # instead of deleting them.
    path = RUNS_DIR / "calibration.json"
    output = json.loads(path.read_text()) if path.exists() else {}

    record = load_sea_ice_extent()
    targets = extent_targets(record)
    print(f"observed extent: {targets.describe()}")
    output["observations"] = {
        "climatology": record.climatology().tolist(),
        "extent_max": targets.maximum,
        "extent_min": targets.minimum,
        "month_max": targets.month_maximum,
        "month_min": targets.month_minimum,
        "first_year": int(record.year.min()),
        "last_year": int(record.year.max()),
        "source": record.source}

    # The coarse scan runs first so that it can seed the Newton refinement.
    seeds = None
    if not arguments.skip_scan:
        print("coarse search over the surface energy budget for each scale")
        output["feasibility"] = seeds = feasibility_scan()
    if not arguments.skip_column:
        print("refining the calibration of each configuration")
        output["column"] = calibrate_column(seeds)
    if not arguments.skip_spatial:
        print("spatial calibration against the satellite record")
        output["spatial"] = calibrate_spatial(targets,
                                              spacing_km=arguments.spacing)

    path.write_text(json.dumps(output, indent=1))
    print(f"written {path}")


if __name__ == "__main__":
    main()
