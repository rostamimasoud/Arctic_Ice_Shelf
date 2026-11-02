"""Stage 14: bifurcation analysis of the spatially resolved model.

The spatially resolved model has so far been integrated only forward in time, so
its threshold is the one claim in the study not backed by a Floquet analysis.
This stage closes that gap.  The one-year stroboscopic map of the full field is
advanced by the *physical* model (twelve monthly integrations), and the leading
Floquet multiplier of that map is tracked by matrix-free power iteration, each
Jacobian-vector product estimated by a finite difference.  A multiplier that
approaches one as the greenhouse forcing rises marks the saddle-node fold at
which the perennial-ice annual cycle loses stability, the spatial analogue of the
fold recovered in the column model.

Because the map is iterated forward only in the spin-up, the stage resolves the
two attracting annual cycles (perennial ice and seasonally ice-free) by
integrating each from a cold and a warm initial field respectively.  A branch is
reported where its multiplier approaches one.  The result is written to
``runs/spatial_bifurcation.json``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import RUNS_DIR, SEED, ensure_dirs
from arcticgnn.models.train import set_threads
from arcticgnn.physics.seaice import SeaIceParams
from arcticgnn.spatial.grid import build_graph, build_grid
from arcticgnn.spatial.model import SpatialArcticModel


def load_physical_model(calibration: dict) -> SpatialArcticModel:
    """Reconstruct the spatially resolved model from the calibration stage."""
    spatial = calibration["spatial"]
    column = SeaIceParams().replace(
        longwave_offset=float(spatial["longwave_offset"]),
        longwave_annual=float(spatial["longwave_annual"]),
        albedo_scale=0.5, forcing=0.0)
    grid = build_grid(100.0)
    graph = build_graph(grid)
    return SpatialArcticModel(grid, graph, column=column), grid


def annual_map(model: SpatialArcticModel, flat: np.ndarray, forcing: float,
               steps_per_year: int = 192) -> np.ndarray:
    """The one-year stroboscopic map ``Phi_F`` of the physical model.

    ``flat`` is the flattened state ``(E..., T_d...)``; the map integrates it for
    one year and returns the flattened state at the same month the year after.
    """
    state = np.stack([flat[:model.grid.n_cells],
                      flat[model.grid.n_cells:]], axis=-1)
    _, settled = model.with_forcing(forcing).integrate(
        state, years=1.0, steps_per_year=steps_per_year)
    return np.concatenate([settled[-1][:, 0], settled[-1][:, 1]])


def leading_multiplier(model: SpatialArcticModel, flat: np.ndarray,
                       forcing: float, iters: int = 120, step: float = 1e-3
                       ) -> tuple[float, float]:
    """Signed leading Floquet multiplier by finite-difference power iteration.

    Repeatedly applying the monodromy matrix to a vector converges to its
    dominant eigenvector; the Rayleigh quotient returns the multiplier with its
    sign, so a fold (through +1) is told apart from a period doubling (through
    -1).  The Jacobian-vector product is a centred finite difference on the
    annual map, so no dense matrix is ever formed.
    """
    rng = np.random.default_rng(SEED)
    norm = float(np.linalg.norm(flat))
    h = step * (norm + 1e-3)
    v = rng.normal(size=flat.size)
    v = v / np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        wp = (annual_map(model, flat + h * v, forcing)
              - annual_map(model, flat - h * v, forcing)) / (2.0 * h)
        lam = float(v @ wp)
        v = wp / np.linalg.norm(wp)
    return lam, norm


def spin_up(model: SpatialArcticModel, forcing: float, thickness_m: float,
            deep_temperature: float, years: float = 90.0,
            steps_per_year: int = 192) -> tuple[np.ndarray, float]:
    """Integrate to the attracting annual cycle and return its January state."""
    warm = model.with_forcing(forcing)
    init = warm.initial_state(thickness_m=thickness_m,
                              deep_temperature=deep_temperature)
    _, settled = warm.integrate(init, years=years,
                                steps_per_year=steps_per_year)
    flat = np.concatenate([settled[-1][:, 0], settled[-1][:, 1]])
    _, one_year = warm.integrate(settled[-1], years=1.0,
                                 steps_per_year=steps_per_year)
    residual = float(np.linalg.norm((one_year[-1] - settled[-1]).reshape(-1)))
    return flat, residual


def scan_branch(model: SpatialArcticModel, forcings, thickness_m: float,
                deep_temperature: float, label: str, arguments) -> list:
    """Spin up and read the multiplier at each forcing of one stable branch."""
    points = []
    for forcing in forcings:
        flat, residual = spin_up(model, float(forcing), thickness_m,
                                 deep_temperature, arguments.spinup)
        lam, norm = leading_multiplier(model, flat, float(forcing),
                                       arguments.iters, arguments.fd_step)
        thickness = float(np.maximum(0.0, -flat[:model.grid.n_cells]).mean()
                          / 9.7)
        state = np.stack([flat[:model.grid.n_cells],
                          flat[model.grid.n_cells:]], axis=-1)
        sept = float(model.extent_km2(state[:, 0]).min() / 1e6)
        points.append({
            "forcing": float(forcing),
            "mean_thickness_m": thickness,
            "september_extent_Mkm2": sept,
            "residual": residual,
            "leading_multiplier": lam,
            "state_norm": norm,
        })
        print(f"  [{label}] F={forcing:5.2f}  thickness={thickness:4.2f} m  "
              f"sept_extent={sept:5.2f}  lam1={lam:+.4f}  res={residual:.2e}")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold", type=float, nargs="+",
                        default=[2.0, 2.5, 3.0, 3.5, 4.0, 4.2, 4.4, 4.6, 4.8,
                                 5.0, 5.2])
    parser.add_argument("--warm", type=float, nargs="+",
                        default=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 9.0])
    parser.add_argument("--spinup", type=float, default=150.0)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()

    ensure_dirs()
    set_threads(arguments.threads)
    calibration = json.loads((RUNS_DIR / "calibration.json").read_text())
    model, grid = load_physical_model(calibration)
    print(f"spatial model on {grid.n_cells} cells; "
          f"tracking the leading Floquet multiplier of the annual map")

    print("perennial branch (cold start):")
    cold = scan_branch(model, arguments.cold, 2.0, 0.6, "cold", arguments)
    print("seasonally ice-free branch (warm start):")
    warm = scan_branch(model, arguments.warm, 0.05, 1.2, "warm", arguments)

    output = {
        "n_cells": int(grid.n_cells),
        "n_state": 2 * int(grid.n_cells),
        "method": "finite-difference power iteration on the physical annual map",
        "spin_up_years": arguments.spinup,
        "fd_step": arguments.fd_step,
        "cold_branch": cold,
        "warm_branch": warm,
    }
    path = RUNS_DIR / "spatial_bifurcation.json"
    path.write_text(json.dumps(output, indent=1))
    print(f"written {path}")


if __name__ == "__main__":
    main()
