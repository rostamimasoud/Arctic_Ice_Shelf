"""Rate-induced tipping experiments.

A system holding two stable states can be pushed out of the one it occupies by a
forcing that changes too quickly, even when the forcing never reaches the value
that would tip it if applied slowly.  The distinction the experiments here are
built around is:

bifurcation tipping
    the quasi-static threshold, the saddle-node located by continuation.  What
    matters is how far the forcing goes.

rate-induced tipping
    the largest forcing the system tolerates when the forcing is ramped at a
    finite rate.  When that is smaller than the quasi-static threshold, the pace
    of the change has done the tipping and not its size.

Every ramp rate is integrated in the same batch.  The forcing is applied as an
array with one entry per member, so a whole family of ramps costs one pass, and
the bisection for the critical rate needs a handful of passes instead of one per
trial.

The integration continues well past the end of the ramp before the outcome is
recorded.  This is a correctness requirement.  The sub-halocline layer adjusts
over several decades, so a system inspected at the end of a short ramp is still
in mid transient and looks as though it retained its ice whatever the forcing
did.  Scoring at that moment would report no rate-induced tipping anywhere,
cleanly and wrongly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..physics.seaice import (ICE_FREE_THRESHOLD, SeaIceColumn, SeaIceParams)

#: Years integrated after the ramp ends, at least.  Several times the adjustment
#: time of the deep layer.
MINIMUM_RELAXATION_YEARS = 400.0


@dataclass
class RampOutcome:
    """Result of a family of ramps sharing a target but differing in duration."""

    parameter: str
    start: float
    target: float
    duration: np.ndarray          # ramp length of each member, yr
    tipped: np.ndarray            # whether each member lost its perennial ice
    thickness_min: np.ndarray     # annual minimum thickness at the end, m
    rate: np.ndarray              # forcing change per year


@dataclass
class CriticalRate:
    """The ramp duration separating tipped from untipped outcomes."""

    parameter: str
    start: float
    target: float
    duration_critical: float      # slowest ramp that still tips, yr
    duration_safe: float          # fastest ramp that does not tip, yr
    rate_critical: float          # corresponding forcing change per year
    bifurcation: float | None     # quasi-static threshold
    threshold_reduction: float | None   # how far below it this target sits


def ramp_forcing(parameter: str, start: float, target: float,
                 duration: np.ndarray):
    """A forcing callable ramping ``parameter`` over each member's duration.

    The returned value is an array with one entry per member, which is what
    allows every ramp rate to share a single integration.  After its own ramp
    ends a member is held at the target, so the integration can continue long
    enough for it to commit to a branch.
    """
    duration = np.asarray(duration, float)
    span = float(target) - float(start)

    def forcing(t_years: float) -> dict:
        fraction = np.clip(t_years / duration, 0.0, 1.0)
        return {parameter: float(start) + span * fraction}

    return forcing


def run_ramps(params: SeaIceParams, parameter: str, start: float, target: float,
              durations: np.ndarray, relaxation_years: float | None = None,
              steps_per_year: int = 200, spin_up_years: float = 300.0,
              noise: float = 0.0, seed: int | None = None) -> RampOutcome:
    """Ramp ``parameter`` from ``start`` to ``target`` at several rates at once."""
    durations = np.asarray(durations, float).ravel()
    if relaxation_years is None:
        relaxation_years = max(3.0 * float(durations.max()),
                               MINIMUM_RELAXATION_YEARS)

    initial = SeaIceColumn(params.with_control(**{parameter: start}))
    settled = initial.spin_up(initial.initial_state("perennial"),
                              years=spin_up_years, steps_per_year=steps_per_year)
    state = np.repeat(settled[None, :], durations.size, axis=0)

    model = SeaIceColumn(params)
    total = float(durations.max()) + relaxation_years
    _, evolved = model.integrate(
        state, years=total, steps_per_year=steps_per_year,
        forcing=ramp_forcing(parameter, start, target, durations),
        noise=noise, seed=seed)

    # Score on the annual cycle at the target forcing, once every member has
    # finished its ramp and relaxed.
    final = SeaIceColumn(params.with_control(**{parameter: target}))
    _, cycle = final.annual_cycle(evolved[-1], steps_per_year=steps_per_year,
                                  store_every=2)
    minimum = final.thickness(cycle[..., 0]).min(axis=0)

    return RampOutcome(parameter=parameter, start=start, target=target,
                       duration=durations,
                       tipped=minimum < ICE_FREE_THRESHOLD,
                       thickness_min=minimum,
                       rate=(target - start) / durations)


def critical_rate(params: SeaIceParams, parameter: str, start: float,
                  target: float, bounds: tuple = (3.0, 3000.0),
                  tolerance: float = 0.03, max_iterations: int = 24,
                  bifurcation: float | None = None,
                  **kwargs) -> CriticalRate | None:
    """Bisect on the ramp duration to find the critical rate.

    Tipping is assumed monotone in the ramp duration: a fast enough ramp tips
    and a slow enough one does not.  Both ends of the bracket are checked first,
    and ``None`` is returned when it contains no transition, meaning either that
    every rate tips, because the target lies beyond the quasi-static threshold,
    or that none does.
    """
    fast, slow = float(bounds[0]), float(bounds[1])
    ends = run_ramps(params, parameter, start, target,
                     np.array([fast, slow]), **kwargs)
    if not ends.tipped[0] or ends.tipped[1]:
        return None

    for _ in range(max_iterations):
        if (slow - fast) / slow < tolerance:
            break
        # Geometric bisection, because the outcome depends on the order of
        # magnitude of the rate rather than on its absolute value.
        middle = float(np.sqrt(fast * slow))
        if run_ramps(params, parameter, start, target,
                     np.array([middle]), **kwargs).tipped[0]:
            fast = middle
        else:
            slow = middle

    reduction = None
    if bifurcation is not None and bifurcation != start:
        reduction = float((bifurcation - target) / (bifurcation - start))

    return CriticalRate(parameter=parameter, start=start, target=target,
                        duration_critical=fast, duration_safe=slow,
                        rate_critical=(target - start) / slow,
                        bifurcation=bifurcation,
                        threshold_reduction=reduction)


def overshoot(params: SeaIceParams, parameter: str, peak: float,
              rise_years: float = 150.0, hold_years: float = 50.0,
              fall_years: float = 150.0, settle_years: float = 600.0,
              start: float = 0.0, steps_per_year: int = 200,
              spin_up_years: float = 300.0, store_every: int = 20,
              noise: float = 0.0, seed: int | None = None) -> dict:
    """Raise the forcing to a peak, return it, and see whether the ice returns.

    This is the experiment that shows irreversibility as something that happens
    in time rather than as a property of a branch diagram.  The forcing is
    ramped up, held, ramped back to where it started, and then held there while
    the system settles.  Plotted against the forcing, the trajectory traces a
    loop whenever the peak crossed a threshold, and retraces its own path when
    it did not.

    A long settling period is essential.  The layer below the halocline gives up
    its heat over decades, so a system inspected shortly after the forcing
    returns still looks changed whether or not it has actually committed to a
    different state.
    """
    peaks = np.atleast_1d(np.asarray(peak, float))
    total = rise_years + hold_years + fall_years + settle_years

    initial = SeaIceColumn(params.with_control(**{parameter: start}))
    settled = initial.spin_up(initial.initial_state("perennial"),
                              years=spin_up_years, steps_per_year=steps_per_year)
    state = np.repeat(settled[None, :], peaks.size, axis=0)

    def forcing(t_years: float) -> dict:
        if t_years <= rise_years:
            fraction = t_years / rise_years
        elif t_years <= rise_years + hold_years:
            fraction = 1.0
        elif t_years <= rise_years + hold_years + fall_years:
            fraction = 1.0 - (t_years - rise_years - hold_years) / fall_years
        else:
            fraction = 0.0
        return {parameter: start + (peaks - start) * fraction}

    model = SeaIceColumn(params)
    times, trajectory = model.integrate(
        state, years=total, steps_per_year=steps_per_year,
        store_every=store_every, forcing=forcing, noise=noise, seed=seed)

    applied = np.array([forcing(t)[parameter] for t in times])
    thickness = model.thickness(trajectory[..., 0])

    # Whether each member came back, judged once the forcing has returned and
    # the system has had time to settle.
    final = SeaIceColumn(params.with_control(**{parameter: start}))
    _, cycle = final.annual_cycle(trajectory[-1], steps_per_year=steps_per_year,
                                  store_every=2)
    recovered = final.thickness(cycle[..., 0]).min(axis=0) > ICE_FREE_THRESHOLD

    return {"peak": peaks.tolist(),
            "time": times.tolist(),
            "forcing": applied.tolist(),
            "thickness": thickness.tolist(),
            "deep_temperature": trajectory[..., 1].tolist(),
            "recovered": recovered.tolist(),
            "rise_years": rise_years, "hold_years": hold_years,
            "fall_years": fall_years, "settle_years": settle_years}


def safe_operating_boundary(params: SeaIceParams, parameter: str, start: float,
                            targets: np.ndarray, bifurcation: float | None = None,
                            bounds: tuple = (3.0, 3000.0),
                            **kwargs) -> dict:
    """Trace the boundary separating safe trajectories from tipping ones.

    For each target the slowest ramp that still tips is found.  Targets beyond
    the quasi-static threshold tip at any rate and are recorded as having an
    infinite critical duration; targets that no rate can tip are recorded as
    absent.
    """
    targets = np.asarray(targets, float).ravel()
    duration = np.full(targets.size, np.nan)
    rate = np.full(targets.size, np.nan)

    for i, target in enumerate(targets):
        if bifurcation is not None and target >= bifurcation:
            duration[i] = np.inf
            rate[i] = 0.0
            continue
        found = critical_rate(params, parameter, start, float(target),
                              bounds=bounds, bifurcation=bifurcation, **kwargs)
        if found is not None:
            duration[i] = found.duration_safe
            rate[i] = found.rate_critical

    return {"target": targets.tolist(),
            "duration_critical": duration.tolist(),
            "rate_critical": rate.tolist(),
            "bifurcation": bifurcation}
