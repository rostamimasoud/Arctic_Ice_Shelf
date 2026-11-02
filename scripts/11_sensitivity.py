"""Stage 11: how the bistable window depends on the parameters that were prescribed.

The main result varies one parameter, the albedo transition scale, and holds every
configuration to the same observed thickness by refitting the one free quantity of the
surface energy budget.  That design answers whether the transition scale controls
reversibility.  It does not answer whether the answer survives plausible changes in the
quantities that were prescribed and never fitted.  This stage asks that question.

For each prescribed parameter in turn the value is moved to the low and the high end of
its observational uncertainty, the surface flux offset is recalibrated so that the
perturbed configuration still reproduces the observed annual mean thickness of the
central Arctic, and the branch of annual cycles is continued again in greenhouse
forcing.  What is reported is the width of the bistable window and the forcing at which
perennial ice is lost, both at the albedo transition scale that carries the widest window
in the control calibration.

Every perturbed configuration is therefore held to the same observed present as the
control, so a change in the window is attributable to the parameter and not to a
different control climate.  Configurations that no surface energy budget can reconcile
with the observed thickness are reported as infeasible, which is itself a constraint.
"""

from __future__ import annotations

import argparse
import json
import numpy as np

from arcticgnn.config import RUNS_DIR, SEED, ensure_dirs, forcing_to_warming
from arcticgnn.physics.seaice import SeaIceColumn, SeaIceParams
from arcticgnn.physics.calibrate import (calibrate_offset, calibrated_params,
                                         refine_periodic_state)
from arcticgnn.dynsys.problems import SeaIceProblem
from arcticgnn.dynsys.continuation import bifurcation_diagram

FORCING_RANGE = (-6.0, 16.0)

#: The parameters that were prescribed from observations and never fitted, each with the
#: low and high end of a plausible range and a short label for the table.  The ranges are
#: deliberately wide: they are meant to cover the spread in the literature rather than a
#: formal confidence interval, so that a window surviving across one of them is robust.
PERTURBATIONS = (
    ("mixed_layer_depth", "Mixed layer depth", "m", 50.0, 100.0),
    ("deep_layer_depth", "Sub-halocline layer depth", "m", 300.0, 700.0),
    ("ventilation", "Ventilation of the deep layer", "W m-2 K-1", 1.50, 3.00),
    ("mixing_ice", "Mixing under closed ice", "W m-2 K-1", 0.35, 0.80),
    ("mixing_open", "Additional mixing over open water", "W m-2 K-1", 3.50, 7.00),
    ("albedo_ice", "Albedo over ice", "1", 0.63, 0.73),
    ("albedo_ocean", "Albedo over open water", "1", 0.15, 0.25),
    ("ocean_heat_flux", "Background ocean heat flux", "W m-2", 1.00, 3.00),
)


def control_scale(calibration: dict, bifurcation: dict) -> float:
    """The albedo transition scale carrying the widest window in the control run."""
    widths = [(b.get("hysteresis") or {}).get("width", 0.0)
              for b in bifurcation["branches"]]
    return float(bifurcation["branches"][int(np.argmax(widths))]["albedo_scale"])


def window_of(params: SeaIceParams, max_steps: int = 1200) -> dict:
    """Continue the annual cycles of one configuration and measure its window.

    The configurations examined here sit close to their folds, where a spin-up of
    any fixed length leaves the state drifting while appearing settled across one
    year, and the continuation then fails to find its first fixed point.  The
    start is therefore spun up and refined by Newton iteration on the annual map
    before the continuation begins, as in the main bifurcation stage.
    """
    problem = SeaIceProblem(params, parameter="forcing",
                            observable="thickness_min")
    model = SeaIceColumn(params.with_control(forcing=0.0))
    start = model.spin_up(model.initial_state("perennial"), years=600.0,
                          steps_per_year=problem.steps_per_year)
    # The refinement works on a batch, so it returns a two dimensional array
    # while the continuation expects the single state it was given.
    start = np.asarray(refine_periodic_state(
        model, start, steps_per_year=problem.steps_per_year)).reshape(-1)[:2]
    branch, folds, window = bifurcation_diagram(
        problem, start, 0.0, FORCING_RANGE[0], FORCING_RANGE[1],
        ds=0.03, max_steps=max_steps)

    # The tipping threshold is the fold crossed as the forcing rises, which the
    # hysteresis object records as its forward value.  Where no window exists the
    # branch is single valued and there is no threshold to report.
    return {"width": float(window.width) if window is not None else 0.0,
            "width_kelvin": (float(forcing_to_warming(window.width))
                             if window is not None else 0.0),
            "threshold": float(window.p_forward) if window is not None else None,
            "recovery": float(window.p_reverse) if window is not None else None,
            "jump": float(window.jump) if window is not None else 0.0,
            "n_folds": len(folds)}


def evaluate(base: SeaIceParams, scale: float, name: str, value: float) -> dict:
    """Recalibrate one perturbed configuration and measure its window."""
    perturbed = base.replace(**{name: float(value)})
    result = calibrate_offset(perturbed, albedo_scale=scale)
    if not result.converged:
        return {"value": float(value), "feasible": False,
                "residual": float(result.residual)}

    params = calibrated_params(perturbed, scale, result)
    out = window_of(params)
    out.update({"value": float(value), "feasible": True,
                "longwave_offset": float(result.longwave_offset),
                "thickness_max": float(result.thickness_max),
                "thickness_min": float(result.thickness_min),
                "residual": float(result.residual)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None,
                        help="restrict to one parameter name")
    arguments = parser.parse_args()

    ensure_dirs()
    calibration = json.loads((RUNS_DIR / "calibration.json").read_text())
    bifurcation = json.loads((RUNS_DIR / "bifurcation.json").read_text())
    scale = control_scale(calibration, bifurcation)

    base = SeaIceParams()
    control = evaluate(base, scale, "latitude", base.latitude)
    print(f"albedo transition scale {scale:.2f} m; "
          f"control window {control['width']:.2f} W m-2 "
          f"({control['width_kelvin']:.2f} K)")

    output = {"seed": SEED, "albedo_scale": scale, "control": control,
              "parameters": []}

    for name, label, unit, low, high in PERTURBATIONS:
        if arguments.only and name != arguments.only:
            continue
        entry = {"name": name, "label": label, "unit": unit,
                 "default": float(getattr(base, name)), "cases": []}
        for value in (low, high):
            case = evaluate(base, scale, name, value)
            entry["cases"].append(case)
            if case["feasible"]:
                threshold = ("none" if case["threshold"] is None
                             else f"{case['threshold']:.2f}")
                print(f"{label} = {value:g} {unit}: window "
                      f"{case['width']:.2f} W m-2, threshold {threshold}")
            else:
                print(f"{label} = {value:g} {unit}: no budget reproduces the "
                      f"observed thickness (residual {case['residual']:.3f} m)")
        feasible = [c["width"] for c in entry["cases"] if c["feasible"]]
        entry["width_range"] = [min(feasible), max(feasible)] if feasible else None
        entry["bistable_throughout"] = bool(feasible) and min(feasible) > 0.0
        output["parameters"].append(entry)

    kept = [e for e in output["parameters"] if e["width_range"] is not None]
    if kept:
        widths = [w for e in kept for w in e["width_range"]]
        output["summary"] = {
            "control_width": control["width"],
            "width_min": float(min(widths)),
            "width_max": float(max(widths)),
            "bistable_in_all_cases": all(e["bistable_throughout"] for e in kept),
            "n_parameters": len(kept)}
        print(f"\nwindow across every perturbation: "
              f"{min(widths):.2f} to {max(widths):.2f} W m-2; "
              f"bistable throughout: {output['summary']['bistable_in_all_cases']}")

    path = RUNS_DIR / "sensitivity.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
