"""Stage 10: supplementary figures.

These support the main text without belonging in it: the calibration itself,
the prescribed transport of the spatially resolved model, the sensitivity of the
early warning indicators to the choice of filter, and the comparison between
emulator architectures. Conventions match the main figures.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import FIGURE_DIR, RUNS_DIR, ensure_dirs
from arcticgnn.viz import maps, style


def load(name: str) -> dict | None:
    path = RUNS_DIR / name
    if not path.exists():
        print(f"  {name} is missing, skipping the figures that need it")
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# S1: the calibration
# --------------------------------------------------------------------------- #

def figure_calibration(calibration: dict) -> None:
    """What the calibration did, and what it left as a prediction."""
    import matplotlib.pyplot as plt

    import matplotlib as mpl

    column = calibration["column"]
    scan = calibration.get("feasibility")

    style.use_style()
    figure, axes_grid = plt.subplots(1, 3, figsize=(style.WIDTH_DOUBLE, 2.6))
    # The first panel carries a colour bar, so its column needs extra room
    # before the next axis label begins.
    figure.subplots_adjust(wspace=0.62)

    scales = np.array(column["albedo_scale"])
    target = column["target_mean"]

    # (a) mean thickness against the offset, for each transition scale.
    axes = axes_grid[0]
    if scan is not None:
        offsets = np.array(scan["offsets"])
        norm = mpl.colors.Normalize(scales.min(), scales.max())
        for scale, curve in zip(scan["albedo_scale"], scan["curves"]):
            mean = np.array(curve["mean"])
            # Only the perennial branch is shown; below the fold the cover is
            # gone and the relation is no longer a branch at all.
            alive = mean > 0.05
            axes.plot(offsets[alive], mean[alive],
                      color=style.SEQUENTIAL(norm(scale)), linewidth=0.9)
        axes.axhline(target, color=style.SOURCE["observations"],
                     linewidth=0.9, linestyle="--")
        axes.annotate("Observed mean", xy=(offsets.min(), target),
                      xytext=(2, 5), textcoords="offset points", fontsize=6,
                      color=style.INK["secondary"], va="bottom")
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=style.SEQUENTIAL)
        maps.add_colourbar(figure, mappable, axes, "Transition scale (m)",
                           pad=0.03, fraction=0.05)
    axes.set_xlabel("Surface flux offset (W m$^{-2}$)")
    axes.set_ylabel("Annual mean thickness (m)")
    axes.set_yscale("log")
    style.soften_grid(axes)
    style.panel_label(axes, "a")

    # (b) the offset each configuration needed.
    axes = axes_grid[1]
    converged = np.array(column["converged"], bool)
    offset = np.array(column["longwave_offset"])
    axes.plot(scales[converged], offset[converged], color=style.CATEGORICAL[0],
              marker="o", markersize=3.2)
    if (~converged).any():
        axes.plot(scales[~converged], offset[~converged], linestyle="none",
                  marker="x", markersize=5.0, markeredgewidth=1.2,
                  color=style.REGIME["warm"], label="Not reachable")
        axes.legend(loc="upper left", fontsize=5.8)
        span = scales.max() - scales.min()
        axes.set_xlim(scales.min() - 0.05 * span, scales.max() + 0.08 * span)
    axes.set_xlabel("Albedo transition scale (m)")
    axes.set_ylabel("Calibrated offset (W m$^{-2}$)")
    style.soften_grid(axes)
    style.panel_label(axes, "b")

    # (c) the seasonal range, which was predicted and not fitted.
    axes = axes_grid[2]
    high = np.array(column["thickness_max"])
    low = np.array(column["thickness_min"])
    axes.fill_between(scales[converged], low[converged], high[converged],
                      color=style.REGIME["cold"], alpha=0.25,
                      label="Modelled range")
    axes.plot(scales[converged], high[converged], color=style.CATEGORICAL[0],
              marker="o", markersize=2.6)
    axes.plot(scales[converged], low[converged], color=style.CATEGORICAL[0],
              marker="o", markersize=2.6)
    axes.axhline(column["target_max"], color=style.SOURCE["observations"],
                 linewidth=0.9, linestyle="--")
    axes.axhline(column["target_min"], color=style.SOURCE["observations"],
                 linewidth=0.9, linestyle="--", label="Observed range")
    axes.axhline(target, color=style.SOURCE["observations"], linewidth=0.9,
                 linestyle=":", label="Observed mean, the target")
    axes.set_xlabel("Albedo transition scale (m)")
    axes.set_ylabel("Ice thickness (m)")
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1,
                fontsize=5.5, frameon=False)
    style.soften_grid(axes)
    style.panel_label(axes, "c")

    style.save(figure, FIGURE_DIR / "figureS1_calibration")
    plt.close(figure)
    print("  figure S1 written")


# --------------------------------------------------------------------------- #
# S2: the prescribed transport
# --------------------------------------------------------------------------- #

def figure_transport(calibration: dict) -> None:
    """The circulation and geometry the spatial model was given."""
    import matplotlib.pyplot as plt

    from arcticgnn.physics.seaice import SeaIceParams
    from arcticgnn.spatial.grid import build_graph, build_grid, distance_to_coast
    from arcticgnn.spatial.model import SpatialArcticModel

    spatial = calibration["spatial"]
    grid = build_grid(spatial.get("spacing_km", 100.0))
    graph = build_graph(grid)
    column = SeaIceParams().replace(
        longwave_offset=float(spatial["longwave_offset"]),
        longwave_annual=float(spatial["longwave_annual"]),
        albedo_scale=float(spatial.get("albedo_scale", 0.5)))
    model = SpatialArcticModel(grid, graph, column=column)

    style.use_style()
    figure, axes_grid = plt.subplots(1, 3, figsize=(style.WIDTH_DOUBLE, 2.4))
    figure.subplots_adjust(wspace=0.16)

    # (a) sea ice drift.
    axes = axes_grid[0]
    speed = np.hypot(model.ice_u, model.ice_v)
    image = maps.draw_field(axes, grid, speed, cmap=style.SEQUENTIAL)
    maps.draw_vectors(axes, grid, model.ice_u, model.ice_v, step=7,
                      colour=style.INK["primary"], scale=1.4e4)
    maps.add_colourbar(figure, image, axes, "Ice drift (km yr$^{-1}$)")
    style.panel_label(axes, "a", dx=-0.02, dy=1.05)

    # (b) the sub-halocline circulation and where Atlantic heat enters.
    axes = axes_grid[1]
    image = maps.draw_field(axes, grid, model.nordic, cmap=style.SEQUENTIAL,
                            vmin=0.0, vmax=1.0)
    maps.draw_vectors(axes, grid, model.ocean_u, model.ocean_v, step=7,
                      colour=style.INK["primary"], scale=7e3)
    maps.add_colourbar(figure, image, axes, "Atlantic inflow weight")
    style.panel_label(axes, "b", dx=-0.02, dy=1.05)

    # (c) distance to the coast, which tapers the drift.
    axes = axes_grid[2]
    image = maps.draw_field(axes, grid, distance_to_coast(grid),
                            cmap=style.SEQUENTIAL)
    maps.add_colourbar(figure, image, axes, "Distance to land (cells)")
    style.panel_label(axes, "c", dx=-0.02, dy=1.05)

    style.save(figure, FIGURE_DIR / "figureS2_transport")
    plt.close(figure)
    print("  figure S2 written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default=None)
    arguments = parser.parse_args()

    ensure_dirs()
    calibration = load("calibration.json")
    if calibration and arguments.only in (None, "calibration"):
        figure_calibration(calibration)
    if calibration and arguments.only in (None, "transport"):
        figure_transport(calibration)
    print(f"supplementary figures are in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
