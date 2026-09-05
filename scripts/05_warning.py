"""Stage 5: whether the approach to the threshold can be seen before it arrives.

As a system nears a saddle-node its recovery from disturbance slows, so
fluctuations grow larger and stay correlated for longer, and the distribution of
their increments loses its symmetry in time.  This stage drives the calibrated
column with a slow forcing ramp and additive noise, then asks whether those
signatures appear in the record before the ice is lost.

Two points decide whether the answer means anything.

The indicators are computed on the residual after the slow forced drift has been
removed, because a rising trend in the mean would otherwise masquerade as rising
variance.  The width of the filter is stated explicitly and the sensitivity of
the answer to it is reported, since an indicator that only trends for one choice
of filter is a property of the filter.

Significance is judged against surrogates that share the power spectrum of the
record but carry randomised phases.  Rolling window indicators are strongly
autocorrelated by construction, so the ordinary rank correlation test is far too
permissive and would declare almost any series significant.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import RUNS_DIR, SEED, ensure_dirs
from arcticgnn.dynsys.ews import compute_ews, indicator_sensitivity
from arcticgnn.dynsys.rtipping import ramp_forcing
from arcticgnn.physics.seaice import ICE_FREE_THRESHOLD, SeaIceColumn, SeaIceParams


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


def ramped_record(params: SeaIceParams, target: float, years: float = 900.0,
                  noise: float = 0.9, steps_per_year: int = 200,
                  start: float = 0.0, seed: int = SEED) -> dict:
    """Drive the column slowly towards the threshold and record every year.

    The ramp is slow enough that the system tracks its stable state until the
    threshold is reached, which is what makes any warning genuinely early rather
    than a response to a forcing changing too fast to follow.
    """
    model = SeaIceColumn(params.replace(forcing=start))
    settled = model.spin_up(model.initial_state("perennial"), years=400.0,
                            steps_per_year=steps_per_year)
    model = SeaIceColumn(params)

    # The duration is a scalar here.  Passing an array would make the forcing
    # array valued, which is how a whole family of ramps is integrated at once
    # elsewhere, but here it would give every state a spurious leading axis.
    forcing = ramp_forcing("forcing", start, target, float(years))
    times, states = model.integrate(
        settled, years=years, steps_per_year=steps_per_year,
        store_every=steps_per_year // 12, forcing=forcing,
        noise=noise, seed=seed)

    thickness = model.thickness(states[:, 0])
    applied = np.array([forcing(t)["forcing"] for t in times]).ravel()

    # One value per year: the thinnest ice reached that year, which is the
    # quantity whose loss defines the transition.
    per_year = thickness[:len(thickness) // 12 * 12].reshape(-1, 12)
    annual_minimum = per_year.min(axis=1)
    annual_forcing = applied[:len(applied) // 12 * 12].reshape(-1, 12).mean(axis=1)

    lost = np.where(annual_minimum < ICE_FREE_THRESHOLD)[0]
    return {"time": times.tolist(), "forcing": applied.tolist(),
            "thickness": thickness.tolist(),
            "annual_minimum": annual_minimum.tolist(),
            "annual_forcing": annual_forcing.tolist(),
            "year_of_loss": int(lost[0]) if lost.size else None}


def analyse(record: dict, window: int = 80, bandwidth: int = 40,
            n_surrogates: int = 300, margin: int = 25) -> dict:
    """Compute the indicators on the record up to shortly before the loss.

    The series is cut before the transition itself.  Including the collapse
    would let the jump dominate every statistic and would prove only that a
    large change is visible once it has happened.
    """
    series = np.array(record["annual_minimum"], float)
    loss = record["year_of_loss"]
    end = (loss - margin) if loss is not None else series.size
    if end < 60:
        return {}

    truncated = series[:end]
    # Shrink the rolling window if the record is short, instead of abandoning
    # the member.  A member that tips early still carries information; only the
    # resolution of the indicators suffers.
    window = min(window, max(int(0.45 * truncated.size), 25))
    bandwidth = max(window // 2, 10)

    result = compute_ews(truncated, window=window, bandwidth=bandwidth,
                         n_surrogates=n_surrogates, seed=SEED)

    sensitivity = indicator_sensitivity(
        truncated, windows=(60, 80, 100, 120), bandwidths=(25, 40, 60))

    return {"analysed_years": int(end),
            "year_of_loss": loss,
            "margin": margin,
            "window": window,
            "bandwidth": bandwidth,
            "index": result.index.tolist(),
            "variance": result.variance.tolist(),
            "autocorrelation": result.autocorrelation.tolist(),
            "recovery_rate": result.recovery_rate.tolist(),
            "skewness": result.skewness.tolist(),
            "irreversibility": result.irreversibility.tolist(),
            "residual": result.residual.tolist(),
            "trends": {name: {"tau": t.tau, "p_value": t.p_value,
                              "p_value_naive": t.p_value_naive,
                              "significant": bool(t.significant)}
                       for name, t in result.trends.items()},
            "sensitivity": {f"{w}_{b}": v
                            for (w, b), v in sensitivity.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=1500.0)
    # The noise must be small compared with the distance to the threshold, or
    # the transition is caused by the noise rather than approached by the ramp.
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--members", type=int, default=6)
    arguments = parser.parse_args()

    ensure_dirs()
    calibration = load("calibration.json")
    bifurcation = load("bifurcation.json")

    feasible = [i for i, ok in enumerate(calibration["column"]["converged"]) if ok]
    branches = bifurcation["branches"]
    # The widest window.  A narrow one cannot be approached slowly enough for a
    # warning to be meaningful, because the noise needed to excite the indicators
    # would then be comparable with the distance to the threshold itself.
    chosen = max(range(len(branches)),
                 key=lambda i: (branches[i].get("hysteresis")
                                or {"width": -1.0})["width"])
    params = configuration(calibration, feasible[chosen])

    branch = branches[chosen]
    window = branch.get("hysteresis")
    if window is None:
        raise SystemExit("this configuration has no threshold to approach")
    threshold = float(window["forcing_forward"])
    # The ramp begins below the window and ends a little beyond the threshold.
    # The margin is a fraction of the window rather than of the threshold
    # itself, which for this configuration sits close to zero and would give no
    # overshoot at all under a multiplicative rule.
    start = float(window["forcing_reverse"]) - 0.35 * window["width"]
    target = threshold + 0.20 * window["width"]

    print(f"ramping from {start:.2f} to {target:.2f} W m-2 over "
          f"{arguments.years:.0f} years, past a threshold at {threshold:.2f}, "
          f"with noise of {arguments.noise:.2f} W m-2")

    output = {"threshold": threshold,
              "albedo_scale": float(params.albedo_scale),
              "members": []}

    for member in range(arguments.members):
        record = ramped_record(params, target, years=arguments.years,
                               noise=arguments.noise, start=start,
                               seed=SEED + 17 * member)
        if record["year_of_loss"] is None:
            print(f"  member {member}: the ice was never lost, skipping")
            continue
        indicators = analyse(record)
        if not indicators:
            print(f"  member {member}: tipped too early to analyse, skipping")
            continue
        trends = indicators["trends"]
        print(f"  member {member}: ice lost in year "
              f"{record['year_of_loss']}; "
              + ", ".join(
                  f"{name} tau {t['tau']:+.2f} "
                  f"(p {t['p_value']:.3f}{'*' if t['significant'] else ''})"
                  for name, t in trends.items()))
        output["members"].append({"seed": SEED + 17 * member,
                                  "record": record,
                                  "indicators": indicators})

    if not output["members"]:
        raise SystemExit("no member reached the transition; lengthen the ramp")

    # How often each indicator gave a significant warning across members.
    names = list(output["members"][0]["indicators"]["trends"])
    output["detection_rate"] = {
        name: float(np.mean([m["indicators"]["trends"][name]["significant"]
                             for m in output["members"]]))
        for name in names}
    print("fraction of members in which each indicator warned significantly:")
    for name, rate in output["detection_rate"].items():
        print(f"  {name:16s} {rate:.2f}")

    path = RUNS_DIR / "warning.json"
    path.write_text(json.dumps(output))
    print(f"written {path}")


if __name__ == "__main__":
    main()
