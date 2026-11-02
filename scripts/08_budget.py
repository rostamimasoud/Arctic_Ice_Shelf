"""Stage 8: how much of the warming has already been applied.

The continuation places each threshold on an axis whose zero is the present. That
is the natural axis for the mathematics and the wrong one for judging urgency,
because it says how much forcing remains without saying how much has been spent.
This stage re-references the same thresholds to preindustrial conditions and
computes, for every calibrated configuration, the fraction of the forcing needed
to end perennial ice that has already been applied.

The uncertainty in the present forcing is carried through rather than quoted
once, since it is the largest term in the calculation and the fraction depends on
it non-linearly: the same absolute uncertainty matters far more for a
configuration whose threshold is close than for one whose threshold is distant.

The thickness of the ice under preindustrial forcing is computed as well. It was
never fitted, so it is a prediction that reconstructions of the preindustrial
cover can be tested against.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import (FORCING_PRESENT, FORCING_PRESENT_RANGE,
                              LATENT_ICE, RUNS_DIR, ensure_dirs,
                              forcing_to_warming)
from arcticgnn.physics.calibrate import refine_periodic_state
from arcticgnn.physics.seaice import SeaIceColumn, SeaIceParams


def load(name: str) -> dict:
    path = RUNS_DIR / name
    if not path.exists():
        raise SystemExit(f"run the earlier stages first, {name} is missing")
    return json.loads(path.read_text())


def thickness_at(scales: np.ndarray, offsets: np.ndarray, forcing: float,
                 spin_up_years: float = 300.0, steps_per_year: int = 200
                 ) -> dict:
    """Annual mean, maximum and minimum thickness of every configuration.

    The whole set is integrated as one batch and then put exactly onto its
    periodic state, which matters here because several configurations sit close
    to a fold where relaxation alone never finishes.
    """
    model = SeaIceColumn(SeaIceParams().replace(
        albedo_scale=scales, longwave_offset=offsets,
        forcing=np.full(scales.size, float(forcing))))
    start = np.stack([np.full(scales.size, -3.0 * LATENT_ICE),
                      np.full(scales.size, 0.5)], axis=-1)
    settled = model.spin_up(start, years=spin_up_years,
                            steps_per_year=steps_per_year)
    settled = refine_periodic_state(model, settled,
                                    steps_per_year=steps_per_year)
    _, cycle = model.annual_cycle(settled, steps_per_year=steps_per_year,
                                  store_every=4)
    thickness = model.thickness(cycle[..., 0])
    return {"mean": thickness.mean(axis=0).tolist(),
            "max": thickness.max(axis=0).tolist(),
            "min": thickness.min(axis=0).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spin-up", type=float, default=300.0)
    arguments = parser.parse_args()

    ensure_dirs()
    calibration = load("calibration.json")
    bifurcation = load("bifurcation.json")

    column = calibration["column"]
    feasible = [i for i, ok in enumerate(column["converged"]) if ok]
    branches = bifurcation["branches"]

    scales = np.array([column["albedo_scale"][i] for i in feasible])
    offsets = np.array([column["longwave_offset"][i] for i in feasible])

    print("thickness under preindustrial forcing and in the current period")
    preindustrial = thickness_at(scales, offsets, -FORCING_PRESENT,
                                 spin_up_years=arguments.spin_up)
    present = thickness_at(scales, offsets, 0.0,
                           spin_up_years=arguments.spin_up)

    entries = []
    for index, scale in enumerate(scales):
        window = branches[index].get("hysteresis")
        threshold = (float(window["forcing_forward"]) if window else None)

        entry = {"albedo_scale": float(scale),
                 "threshold_from_present": threshold,
                 "thickness_preindustrial": preindustrial["mean"][index],
                 "thickness_present": present["mean"][index],
                 "thinning": (preindustrial["mean"][index]
                              - present["mean"][index])}

        if threshold is not None:
            span = threshold + FORCING_PRESENT
            entry.update({
                "threshold_from_preindustrial": span,
                "fraction_applied": FORCING_PRESENT / span,
                "remaining": max(threshold, 0.0),
                "remaining_kelvin": forcing_to_warming(max(threshold, 0.0)),
                # The uncertainty in the present forcing shifts both the numerator
                # and the total, so it is propagated rather than added afterwards.
                "fraction_low": (FORCING_PRESENT_RANGE[0]
                                 / (threshold + FORCING_PRESENT_RANGE[0])),
                "fraction_high": (FORCING_PRESENT_RANGE[1]
                                  / (threshold + FORCING_PRESENT_RANGE[1]))})
            print(f"  scale {scale:.2f} m: threshold {span:5.2f} W m-2 above "
                  f"preindustrial, {100 * entry['fraction_applied']:3.0f} per "
                  f"cent applied, {entry['remaining']:.2f} W m-2 remaining "
                  f"({entry['remaining_kelvin']:.2f} K)")
        else:
            print(f"  scale {scale:.2f} m: no threshold within the range "
                  f"examined")
        entries.append(entry)

    with_threshold = [e for e in entries if e.get("fraction_applied")]
    if with_threshold:
        fractions = [e["fraction_applied"] for e in with_threshold]
        thinning = [e["thinning"] for e in entries]
        print(f"across configurations carrying a threshold, "
              f"{100 * min(fractions):.0f} to {100 * max(fractions):.0f} per "
              f"cent of the forcing has been applied")
        print(f"predicted thinning since preindustrial: "
              f"{min(thinning):.2f} to {max(thinning):.2f} m")

    output = {"forcing_present": FORCING_PRESENT,
              "forcing_present_range": list(FORCING_PRESENT_RANGE),
              "entries": entries,
              "preindustrial": preindustrial,
              "present": present}

    path = RUNS_DIR / "budget.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"written {path}")


if __name__ == "__main__":
    main()
