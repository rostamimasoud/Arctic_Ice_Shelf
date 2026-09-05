"""Calibration of the column model to the observed seasonal cycle.

Two parameters of the surface energy budget are not known well enough to be
prescribed: the temperature-independent part of the net upward surface flux and
its seasonal amplitude.  Together they control the mean thickness of the ice and
the size of its seasonal swing, so they can be fixed by requiring the control
integration to reproduce two observed numbers, the seasonal maximum and minimum
thickness.

Why this matters for the result rather than only for realism: the position of
the saddle-node fold depends on the albedo transition scale, which is itself
poorly constrained.  If each choice of that scale were run with the same energy
budget, the configurations would differ in their present-day climate as well as
in their tipping behaviour, and the two effects could not be separated.  Every
configuration is therefore recalibrated to the *same* observed control climate
first.  Differences in tipping behaviour that survive are attributable to the
parameter under study.

The solver is a damped Newton iteration written to work on many configurations
at once.  A finite-difference Jacobian needs the residual at the current point
and at two perturbed points, and because the model parameters may be arrays, all
three evaluations for all configurations are packed into a single batched
integration.  Calibrating six configurations therefore costs about as much as
calibrating one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .seaice import LATENT_ICE, SeaIceColumn, SeaIceParams

#: Observed seasonal maximum and minimum ice thickness of the central Arctic
#: Ocean, in m.
TARGET_THICKNESS_MAX = 2.10
TARGET_THICKNESS_MIN = 1.30

#: Observed annual mean thickness, m.  This single number is the calibration
#: target.  Only one quantity of the surface energy budget is left free, because
#: its seasonal amplitude is already fixed by the observed non-solar fluxes, so
#: there is one target to match and the seasonal range of thickness that results
#: becomes a prediction to be tested rather than something fitted.
TARGET_THICKNESS_MEAN = 0.5 * (TARGET_THICKNESS_MAX + TARGET_THICKNESS_MIN)


@dataclass
class CalibrationResult:
    """Outcome of calibrating one configuration."""

    longwave_offset: float
    longwave_annual: float
    thickness_max: float
    thickness_min: float
    residual: float          # largest absolute thickness error, m
    converged: bool
    iterations: int


# --------------------------------------------------------------------------- #
# Forward evaluation
# --------------------------------------------------------------------------- #

def seasonal_extremes(params: SeaIceParams, offset: np.ndarray,
                      annual: np.ndarray, spin_up_years: float = 400.0,
                      steps_per_year: int = 200,
                      store_every: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Seasonal maximum and minimum thickness for each configuration.

    ``offset`` and ``annual`` are flat arrays of equal length; the model is run
    once with array-valued parameters, so the whole set is integrated together.
    """
    offset = np.asarray(offset, float).ravel()
    annual = np.asarray(annual, float).ravel()
    if offset.shape != annual.shape:
        raise ValueError("offset and annual must have the same shape")

    model = SeaIceColumn(params.replace(longwave_offset=offset,
                                        longwave_annual=annual,
                                        forcing=np.zeros_like(offset)))
    start = np.stack([np.full_like(offset, -3.0 * LATENT_ICE),
                      np.full_like(offset, 0.5)], axis=-1)
    settled = model.spin_up(start, years=spin_up_years,
                            steps_per_year=steps_per_year)
    _, trajectory = model.annual_cycle(settled, steps_per_year=steps_per_year,
                                       store_every=store_every)
    thickness = model.thickness(trajectory[..., 0])
    return thickness.max(axis=0), thickness.min(axis=0)


def _residual(params: SeaIceParams, unknowns: np.ndarray,
              target_max: float, target_min: float,
              **kwargs) -> np.ndarray:
    """Thickness errors, shape ``(n, 2)``."""
    h_max, h_min = seasonal_extremes(params, unknowns[:, 0], unknowns[:, 1],
                                     **kwargs)
    return np.stack([h_max - target_max, h_min - target_min], axis=-1)


# --------------------------------------------------------------------------- #
# Batched Newton solve
# --------------------------------------------------------------------------- #

def calibrate(params: SeaIceParams,
              albedo_scales: np.ndarray | None = None,
              offset_guess: float = 80.0, annual_guess: float = -51.0,
              target_max: float = TARGET_THICKNESS_MAX,
              target_min: float = TARGET_THICKNESS_MIN,
              tolerance: float = 5e-3, max_iterations: int = 25,
              step_offset: float = 0.05, step_annual: float = 0.05,
              damping: float = 1.0, max_step: float = 3.0,
              verbose: bool = False,
              **kwargs) -> list[CalibrationResult]:
    """Calibrate one configuration per entry of ``albedo_scales``.

    Parameters
    ----------
    params
        Baseline parameters.  Its ``albedo_scale`` is replaced by each entry of
        ``albedo_scales``; pass ``None`` to calibrate the baseline alone.
    offset_guess, annual_guess
        Starting values for the two unknowns.
    target_max, target_min
        Observed seasonal maximum and minimum thickness, m.
    tolerance
        Convergence is declared when both thickness errors fall below this, m.
    step_offset, step_annual
        Finite-difference steps for the Jacobian, W m^-2.
    damping
        Newton steps are multiplied by this factor.  Reduce below one if the
        iteration overshoots across the fold, where the residual is
        discontinuous and a full step can jump onto the seasonal branch.

    Returns
    -------
    list of CalibrationResult
        One entry per configuration, in the order given.
    """
    scales = (np.asarray(albedo_scales, float).ravel()
              if albedo_scales is not None
              else np.array([params.albedo_scale], float))
    n = scales.size

    # Configurations are stacked along the batch axis; every quantity below
    # carries one row per configuration.
    unknowns = np.stack([np.full(n, float(offset_guess)),
                         np.full(n, float(annual_guess))], axis=-1)
    steps = np.array([step_offset, step_annual])

    def evaluate(u_all: np.ndarray) -> np.ndarray:
        """Residual for a stack of ``(n, 2)`` blocks laid end to end."""
        blocks = u_all.reshape(-1, 2)
        tiled = params.replace(albedo_scale=np.tile(scales, u_all.shape[0] // n))
        return _residual(tiled, blocks, target_max, target_min, **kwargs)

    # A configuration is abandoned once its residual stops responding to either
    # unknown.  That happens when the iterate has wandered onto the branch with
    # no summer ice, where the seasonal extremes are identically zero and the
    # Jacobian vanishes, so Newton has no direction to move in.  Freezing such a
    # configuration and reporting it as unconverged is the honest outcome: it
    # means the target control climate is unreachable for that parameter value.
    active = np.ones(n, bool)
    stalled_tolerance = 1e-8
    iterations = 0
    residual = np.full((n, 2), np.nan)

    for iterations in range(1, max_iterations + 1):
        # Base point and both perturbation directions travel in one batched
        # integration, so an iteration costs a single pass over the ensemble.
        stacked = np.concatenate(
            [unknowns] + [unknowns + np.eye(2)[j] * steps[j] for j in range(2)],
            axis=0)
        values = evaluate(stacked)
        residual = values[:n]

        jacobian = np.empty((n, 2, 2))
        for j in range(2):
            block = values[(j + 1) * n:(j + 2) * n]
            jacobian[:, :, j] = (block - residual) / steps[j]

        converged = np.all(np.abs(residual) < tolerance, axis=1)
        active &= ~converged
        active &= np.abs(jacobian).reshape(n, -1).max(axis=1) > stalled_tolerance
        if not active.any():
            break

        for i in np.where(active)[0]:
            try:
                step = damping * np.linalg.solve(jacobian[i], residual[i])
            except np.linalg.LinAlgError:
                active[i] = False
                continue
            # Limit how far one iteration may travel.  The residual is
            # discontinuous at the fold, where summer ice vanishes, and an
            # unrestrained Newton step computed from the Jacobian on the
            # perennial branch readily overshoots across it.  Once on the far
            # side the extremes are identically zero, the Jacobian vanishes and
            # the configuration can never come back.
            longest = np.abs(step).max()
            if longest > max_step:
                step *= max_step / longest
            unknowns[i] -= step

        if verbose:
            worst = np.abs(residual).max(axis=1)
            print(f"  iteration {iterations:2d}  "
                  f"max error {worst.max():.4f} m  "
                  f"({int(active.sum())} of {n} still moving)")

    h_max, h_min = seasonal_extremes(
        params.replace(albedo_scale=scales), unknowns[:, 0], unknowns[:, 1],
        **kwargs)

    return [CalibrationResult(
        longwave_offset=float(unknowns[i, 0]),
        longwave_annual=float(unknowns[i, 1]),
        thickness_max=float(h_max[i]),
        thickness_min=float(h_min[i]),
        residual=float(max(abs(h_max[i] - target_max),
                           abs(h_min[i] - target_min))),
        converged=bool(max(abs(h_max[i] - target_max),
                           abs(h_min[i] - target_min)) < tolerance),
        iterations=iterations,
    ) for i in range(n)]


def refine_periodic_state(model: SeaIceColumn, state: np.ndarray,
                          steps_per_year: int = 200, n_iterations: int = 8,
                          tolerance: float = 1e-8,
                          step: float = 1e-5) -> np.ndarray:
    """Newton-solve a batch of states onto exact periodic states.

    Solves the annual map for its fixed point, so that the state returned
    repeats itself after a year to the tolerance given.

    A long spin-up is not a substitute for this and cannot be made one by
    lengthening it.  Approaching a fold the leading Floquet multiplier tends to
    one, so relaxation slows without limit, and a spin-up of any fixed length
    leaves the state still drifting.  The drift is invisible in a single annual
    cycle, so the diagnostics look settled while describing a state the model
    does not actually hold.  We found the calibration reporting perennial ice
    where the converged state has none, purely from this.

    Every Jacobian for the whole batch comes from three batched integrations, so
    the refinement costs a few years of integration instead of the thousands
    that slow relaxation would need.
    """
    state = np.atleast_2d(np.asarray(state, float)).copy()
    n_batch = state.shape[0]
    scale = np.array([LATENT_ICE, 1.0])

    for _ in range(n_iterations):
        scaled = state / scale
        residual = (model.annual_map(state, steps_per_year=steps_per_year)
                    / scale - scaled)
        if np.abs(residual).max() < tolerance:
            break

        # Both perturbation directions for the whole batch in one call.
        jacobian = np.empty((n_batch, 2, 2))
        for j in range(2):
            shifted = scaled.copy()
            shifted[:, j] += step
            moved = (model.annual_map(shifted * scale,
                                      steps_per_year=steps_per_year) / scale
                     - shifted)
            jacobian[:, :, j] = (moved - residual) / step

        for i in range(n_batch):
            try:
                scaled[i] -= np.linalg.solve(jacobian[i], residual[i])
            except np.linalg.LinAlgError:
                continue
        state = scaled * scale

    return state


def mean_thickness(params: SeaIceParams, offset: np.ndarray,
                   spin_up_years: float = 300.0, steps_per_year: int = 200,
                   store_every: int = 2, refine: bool = True) -> tuple:
    """Annual mean, maximum and minimum thickness for each offset.

    The seasonal amplitude of the surface flux is taken from ``params`` and is
    not varied, so the whole set of offsets integrates as one batch.  The
    spin-up brings each column into the basin of its attractor and the Newton
    refinement then puts it exactly on the periodic state, which matters near a
    fold where relaxation alone never finishes.
    """
    offset = np.asarray(offset, float).ravel()
    model = SeaIceColumn(params.replace(longwave_offset=offset,
                                        forcing=np.zeros_like(offset)))
    start = np.stack([np.full_like(offset, -3.0 * LATENT_ICE),
                      np.full_like(offset, 0.5)], axis=-1)
    settled = model.spin_up(start, years=spin_up_years,
                            steps_per_year=steps_per_year)
    if refine:
        settled = refine_periodic_state(model, settled,
                                        steps_per_year=steps_per_year)
    _, trajectory = model.annual_cycle(settled, steps_per_year=steps_per_year,
                                       store_every=store_every)
    thickness = model.thickness(trajectory[..., 0])
    return (thickness.mean(axis=0), thickness.max(axis=0),
            thickness.min(axis=0))


def calibrate_offset(params: SeaIceParams, albedo_scale: float,
                     target_mean: float = TARGET_THICKNESS_MEAN,
                     bracket: tuple = (52.0, 104.0), n_coarse: int = 27,
                     tolerance: float = 5e-3, max_iterations: int = 30,
                     verbose: bool = False, **kwargs) -> CalibrationResult:
    """Find the surface flux offset reproducing the observed mean thickness.

    Thicker ice always requires a larger net upward flux, so the mean thickness
    increases monotonically with the offset wherever perennial ice exists.  The
    search exploits that: a coarse sweep brackets the target, then bisection
    closes on it.  Bisection is used in preference to a Newton iteration because
    the response is flat on the far side of the fold, where no summer ice
    survives, and a derivative based method started or landing there cannot
    recover.
    """
    scaled = params.replace(albedo_scale=float(albedo_scale))

    grid = np.linspace(bracket[0], bracket[1], n_coarse)
    mean, _, _ = mean_thickness(scaled, grid, **kwargs)

    # Only offsets holding perennial ice are candidates; below the fold the
    # thickness collapses and the relation is no longer monotone.
    usable = np.where(mean > 0.05)[0]
    if usable.size == 0:
        return CalibrationResult(float("nan"), float(scaled.longwave_annual),
                                 float("nan"), float("nan"), float("inf"),
                                 False, 0)

    below = [i for i in usable if mean[i] <= target_mean]
    above = [i for i in usable if mean[i] >= target_mean]
    if not below or not above:
        # The target lies in the gap the fold opens.  Widening the search does
        # not help and neither does a finer grid: the branch that survives the
        # summer simply does not reach this thickness at this albedo transition
        # scale, so the ice is either appreciably thicker than observed or gone
        # by late summer.  Reporting the closest reachable state and marking it
        # unconverged is the correct outcome, and it is a statement about the
        # parameter rather than about the search.
        best = usable[int(np.argmin(np.abs(mean[usable] - target_mean)))]
        _, high, low = mean_thickness(scaled, np.array([grid[best]]), **kwargs)
        return CalibrationResult(float(grid[best]),
                                 float(scaled.longwave_annual),
                                 float(high[0]), float(low[0]),
                                 float(abs(mean[best] - target_mean)),
                                 False, 0)

    left, right = float(grid[max(below)]), float(grid[min(above)])
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        middle = 0.5 * (left + right)
        mean_mid, high, low = mean_thickness(scaled, np.array([middle]), **kwargs)
        error = float(mean_mid[0] - target_mean)
        if verbose:
            print(f"    offset {middle:7.3f} -> mean thickness "
                  f"{mean_mid[0]:.3f} m (error {error:+.4f})")
        if abs(error) < tolerance:
            break
        if error < 0.0:
            left = middle
        else:
            right = middle

    return CalibrationResult(longwave_offset=float(middle),
                             longwave_annual=float(scaled.longwave_annual),
                             thickness_max=float(high[0]),
                             thickness_min=float(low[0]),
                             residual=float(abs(error)),
                             converged=bool(abs(error) < tolerance),
                             iterations=iterations)


def calibrate_sequence(params: SeaIceParams, albedo_scales: np.ndarray,
                       offset_guess: float = 61.0, annual_guess: float = -28.0,
                       verbose: bool = False, **kwargs) -> list[CalibrationResult]:
    """Calibrate a sequence of configurations, warm starting along it.

    Each configuration begins from the solution found for the previous one.
    This matters because the fold moves as the albedo transition scale changes,
    and a guess that sits on the perennial branch for one value of the scale can
    lie beyond the fold for the next.  Starting every configuration from a fixed
    guess therefore leaves the larger scales stranded on the branch with no
    summer ice, where the calibration has nothing to work with.  Stepping along
    the sequence keeps every solve inside the basin it needs.

    A configuration that still fails to converge is reported as such.  That is a
    result and not a numerical nuisance: it means no surface energy budget
    reproduces the observed present-day seasonal cycle at that value of the
    parameter, so observations of the current Arctic exclude it.
    """
    scales = np.asarray(albedo_scales, float).ravel()
    results: list[CalibrationResult] = []
    offset, annual = float(offset_guess), float(annual_guess)

    for scale in scales:
        if verbose:
            print(f"albedo transition scale {scale:.2f} m, "
                  f"starting from ({offset:.2f}, {annual:.2f})")
        result = calibrate(params, albedo_scales=np.array([scale]),
                           offset_guess=offset, annual_guess=annual,
                           verbose=verbose, **kwargs)[0]
        results.append(result)
        if result.converged:
            offset, annual = result.longwave_offset, result.longwave_annual
    return results


def calibrated_params(params: SeaIceParams, albedo_scale: float,
                      result: CalibrationResult) -> SeaIceParams:
    """Apply a calibration result to a parameter set."""
    return params.replace(albedo_scale=float(albedo_scale),
                          longwave_offset=result.longwave_offset,
                          longwave_annual=result.longwave_annual,
                          forcing=0.0)
