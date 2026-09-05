"""Stage 3: how fast the forcing changes, not only how far it goes.

The continuation stage established where the perennial ice branch ends when the
forcing is changed slowly enough for the system to keep up.  This stage asks the
separate question of whether a forcing that stops short of that threshold can
still remove the ice by arriving too quickly.

The experiment is a family of ramps.  The forcing is raised from its present-day
value to a target below the quasi-static threshold, over durations spanning
three orders of magnitude, and each run is then held at the target long enough
for the system to settle.  Where a short ramp loses the ice and a long ramp to
the same target does not, the pace of the change caused the loss.

The boundary between the two outcomes is traced across a range of targets, which
gives the set of trajectories that stay safe: for every target short of the
threshold there is a slowest ramp that still tips, and anything slower is
survivable.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import RUNS_DIR, ensure_dirs, forcing_to_warming
from arcticgnn.dynsys.rtipping import (overshoot, run_ramps,
                                       safe_operating_boundary)
from arcticgnn.physics.seaice import SeaIceParams

#: Ramp durations examined, years.
DURATIONS = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0])


def load(name: str) -> dict:
    path = RUNS_DIR / name
    if not path.exists():
        raise SystemExit(f"run the earlier stages first, {name} is missing")
    return json.loads(path.read_text())


def configuration(calibration: dict, index: int) -> SeaIceParams:
    column = calibration["column"]
    return SeaIceParams().replace(
        albedo_scale=float(column["albedo_scale"][index]),
        longwave_offset=float(column["longwave_offset"][index]),
        longwave_annual=float(column["longwave_annual"][index]),
        forcing=0.0)


def forward_threshold(branch: dict) -> float | None:
    """The quasi-static forcing at which the perennial branch ends."""
    window = branch.get("hysteresis")
    if window is not None:
        return float(window["forcing_forward"])
    forward = [f["forcing"] for f in branch["folds"]
               if f["direction"] == "upper_to_lower"]
    return float(max(forward)) if forward else None


def ramp_family(params: SeaIceParams, threshold: float,
                fraction: float = 0.88, verbose: bool = True) -> dict:
    """Ramp to a target short of the threshold at a range of rates."""
    target = fraction * threshold
    outcome = run_ramps(params, "forcing", 0.0, target, DURATIONS)
    if verbose:
        print(f"  target {target:.2f} W m-2, which is {fraction:.0%} of the "
              f"quasi-static threshold of {threshold:.2f}")
        for duration, tipped, thickness in zip(outcome.duration, outcome.tipped,
                                               outcome.thickness_min):
            print(f"    ramp over {duration:7.1f} yr -> minimum thickness "
                  f"{thickness:5.2f} m  {'lost' if tipped else 'retained'}")
    return {"target": float(target), "fraction": float(fraction),
            "threshold": float(threshold),
            "duration": outcome.duration.tolist(),
            "tipped": outcome.tipped.tolist(),
            "thickness_min": outcome.thickness_min.tolist(),
            "rate": outcome.rate.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=int, default=None,
                        help="index into the calibrated configurations")
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--skip-overshoot", action="store_true")
    arguments = parser.parse_args()

    ensure_dirs()
    calibration = load("calibration.json")
    bifurcation = load("bifurcation.json")

    feasible = [i for i, ok in enumerate(calibration["column"]["converged"]) if ok]
    branches = bifurcation["branches"]

    chosen = (arguments.configuration if arguments.configuration is not None
              else len(feasible) // 2)
    params = configuration(calibration, feasible[chosen])
    threshold = forward_threshold(branches[chosen])
    if threshold is None:
        raise SystemExit("this configuration has no quasi-static threshold to "
                         "compare a ramp against")

    print(f"albedo transition scale {params.albedo_scale:.2f} m, "
          f"quasi-static threshold {threshold:.2f} W m-2 "
          f"({forcing_to_warming(threshold):.2f} K equivalent warming)")

    output = {"albedo_scale": float(params.albedo_scale),
              "threshold": float(threshold),
              "threshold_kelvin": forcing_to_warming(threshold)}

    print("ramps to a target short of the threshold")
    output["ramps"] = [ramp_family(params, threshold, fraction)
                       for fraction in (0.80, 0.88, 0.94)]

    if not arguments.skip_boundary:
        print("tracing the boundary between safe and tipping trajectories")
        targets = np.linspace(0.45 * threshold, 0.99 * threshold, 12)
        output["boundary"] = safe_operating_boundary(
            params, "forcing", 0.0, targets, bifurcation=threshold)
        for target, duration in zip(output["boundary"]["target"],
                                    output["boundary"]["duration_critical"]):
            print(f"  target {target:6.2f} W m-2  slowest tipping ramp "
                  f"{duration if np.isfinite(duration) else float('inf'):8.1f} yr")

    if not arguments.skip_overshoot:
        print("raising the forcing past the threshold and bringing it back")
        peaks = threshold * np.array([0.7, 0.95, 1.05, 1.3, 1.8])
        result = overshoot(params, "forcing", peaks)
        output["overshoot"] = result
        for peak, back in zip(result["peak"], result["recovered"]):
            print(f"  peak {peak:6.2f} W m-2 "
                  f"({peak / threshold:.0%} of the threshold): "
                  f"{'ice returned' if back else 'ice did not return'}")

    path = RUNS_DIR / "tipping.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"written {path}")


if __name__ == "__main__":
    main()
