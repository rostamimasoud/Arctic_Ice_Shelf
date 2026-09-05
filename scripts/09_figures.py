"""Stage 9: build the manuscript figures from the stored results.

Every panel is drawn from the output of the model runs or from the satellite
record.  Nothing here is a diagram of how something is supposed to work.

The conventions follow the requirements of the journals this work is aimed at.
No panel carries a title inside the image, because titles belong in the legend.
Panels are labelled with bold lower case letters.  Output is vector, with a high
resolution raster copy alongside for convenience.  Axis labels carry units, and
colour is never the only thing distinguishing one series from another.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from arcticgnn.config import (FIGURE_DIR, RUNS_DIR, ensure_dirs,
                              forcing_to_warming)
from arcticgnn.viz import maps, style


def load(name: str) -> dict | None:
    path = RUNS_DIR / name
    if not path.exists():
        print(f"  {name} is missing, skipping the figures that need it")
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Figure 1: the present-day Arctic
# --------------------------------------------------------------------------- #

def figure_present_day(calibration: dict) -> None:
    """Modelled control state validated against the satellite record.

    The observed edge is the published median position of the 15 per cent
    concentration contour, which is the same threshold used to define extent, so
    the line can be laid over the modelled field without any conversion.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from arcticgnn.data.observations import load_median_ice_edge
    from arcticgnn.physics.seaice import SeaIceParams
    from arcticgnn.spatial.grid import build_graph, build_grid
    from arcticgnn.spatial.model import SpatialArcticModel

    spatial = calibration["spatial"]
    grid = build_grid(spatial.get("spacing_km", 100.0))
    graph = build_graph(grid)
    column = SeaIceParams().replace(
        longwave_offset=float(spatial["longwave_offset"]),
        longwave_annual=float(spatial["longwave_annual"]),
        albedo_scale=float(spatial.get("albedo_scale", 0.5)))
    model = SpatialArcticModel(grid, graph, column=column)

    _, settled = model.integrate(model.initial_state(), years=110.0,
                                 steps_per_year=192)
    times, cycle = model.integrate(settled[-1], years=1.0, steps_per_year=192,
                                   store_every=4)
    thickness = model.thickness(cycle[:, :, 0])
    concentration = model.concentration(cycle[:, :, 0])
    extent = model.extent_km2(cycle[:, :, 0]) / 1e6
    month_index = np.clip(((times % 1.0) * 12).astype(int), 0, 11)

    modelled = np.array([extent[month_index == m].mean()
                         if (month_index == m).any() else np.nan
                         for m in range(12)])
    observed = np.array(calibration["observations"]["climatology"], float)
    reach = float(np.abs(grid.x).max()) + grid.spacing_km

    style.use_style()
    figure = plt.figure(figsize=(style.WIDTH_DOUBLE, 5.15))
    # A dedicated strip between the two rows carries the shared legend.  Placing
    # it inside a map instead would cover the ice in the Sea of Okhotsk, and
    # floating it in the gap by hand collides with the colour bar labels.
    spec = figure.add_gridspec(3, 2, hspace=0.30, wspace=0.26,
                               height_ratios=[1.28, 0.09, 1.0])

    # (a) and (b): the two seasonal extremes, each with the observed edge.
    limit = float(np.percentile(thickness, 99.0))
    for position, month, letter in (((0, 0), 3, "a"), ((0, 1), 9, "b")):
        axes = figure.add_subplot(spec[position])
        chosen = int(np.argmax(month_index == month - 1))
        image = maps.draw_field(axes, grid, thickness[chosen],
                                cmap=style.SEQUENTIAL, vmin=0.0, vmax=limit)
        maps.draw_contour(axes, grid, concentration[chosen], [0.15],
                          style.SOURCE["emulator"], linewidth=1.0)
        maps.draw_ice_edge(axes, load_median_ice_edge(month), reach,
                           style.SOURCE["observations"], linewidth=1.0,
                           linestyle="--")
        maps.draw_graticule(axes, reach)
        bar = maps.add_colourbar(figure, image, axes, "Ice thickness (m)")
        # A rule on the colour scale at the thickness separating the thin
        # seasonal cover from the thicker perennial pack.
        bar.ax.axhline(0.5, color=style.INK["primary"], linewidth=0.8)
        style.panel_label(axes, letter, dx=-0.02, dy=1.05)

    handles = [Line2D([], [], color=style.SOURCE["emulator"], linewidth=1.1,
                      label="Modelled ice edge"),
               Line2D([], [], color=style.SOURCE["observations"], linewidth=1.1,
                      linestyle="--", label="Observed median ice edge")]
    strip = figure.add_subplot(spec[1, :])
    strip.axis("off")
    strip.legend(handles=handles, loc="center", ncol=2, fontsize=6.4,
                 frameon=False, handlelength=2.0, columnspacing=2.2)

    # (c) seasonal cycle of extent.
    axes = figure.add_subplot(spec[2, 0])
    months = np.arange(1, 13)
    axes.plot(months, observed, color=style.SOURCE["observations"],
              marker="o", markersize=2.6, label="Satellite record")
    axes.plot(months, modelled, color=style.SOURCE["emulator"],
              marker="s", markersize=2.6, linestyle="--", label="Model")
    axes.set_xlabel("Month")
    axes.set_ylabel("Sea ice extent (10$^6$ km$^2$)")
    axes.set_xticks([1, 3, 5, 7, 9, 11])
    axes.set_xlim(0.5, 12.5)
    axes.set_ylim(0.0, max(observed.max(), np.nanmax(modelled)) * 1.18)
    axes.legend(loc="lower left")
    style.soften_grid(axes)
    style.panel_label(axes, "c")

    # (d) modelled against observed, month by month.
    axes = figure.add_subplot(spec[2, 1])
    good = np.isfinite(modelled) & np.isfinite(observed)
    span = [0.0, max(observed[good].max(), modelled[good].max()) * 1.08]
    axes.plot(span, span, color=style.INK["muted"], linewidth=0.8,
              linestyle=":", label="One to one")
    axes.scatter(observed[good], modelled[good], s=16, zorder=4,
                 color=style.SOURCE["emulator"],
                 edgecolor=style.INK["surface"], linewidth=0.3)
    # Month is cyclic, so a sequential colour scale would imply an order that
    # does not exist; the two seasonal extremes are labelled directly instead.
    for month, offset in ((3, (7, -2)), (9, (7, 2))):
        style.direct_label(axes, observed[month - 1], modelled[month - 1],
                           ("March" if month == 3 else "September"),
                           style.SOURCE["emulator"], dx=offset[0])
    ratio = float(np.median(modelled[good] / np.maximum(observed[good], 1e-9)))
    axes.annotate(f"Median ratio {ratio:.2f}", xy=(0.05, 0.92),
                  xycoords="axes fraction", fontsize=6,
                  color=style.INK["secondary"], va="top")
    axes.set_xlabel("Observed extent (10$^6$ km$^2$)")
    axes.set_ylabel("Modelled extent (10$^6$ km$^2$)")
    axes.set_xlim(span)
    axes.set_ylim(span)
    axes.legend(loc="lower right")
    style.soften_grid(axes)
    style.panel_label(axes, "d")

    style.save(figure, FIGURE_DIR / "figure1_validation")
    plt.close(figure)

    print(f"  figure 1 written; modelled March {modelled[2]:.2f} against "
          f"observed {observed[2]:.2f}, September {modelled[8]:.2f} against "
          f"{observed[8]:.2f} million km2, median ratio {ratio:.2f}")


# --------------------------------------------------------------------------- #
# Figure 2: bifurcation structure
# --------------------------------------------------------------------------- #

def figure_bifurcation(bifurcation: dict) -> None:
    """Branches of annual cycles, their folds, and the dependence on albedo."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    branches = bifurcation["branches"]
    scales = bifurcation["albedo_scale"]

    # The example panel uses the widest window found, since that is where the
    # structure is legible; the panel beside it shows the whole family.
    widest = max(range(len(branches)),
                 key=lambda i: (branches[i].get("hysteresis")
                                or {"width": -1.0})["width"])

    style.use_style()
    figure, axes_grid = plt.subplots(2, 2, figsize=(style.WIDTH_DOUBLE, 4.7))
    # The lower left panel carries a second vertical axis, so the columns need
    # more room between them than the default.
    figure.subplots_adjust(hspace=0.40, wspace=0.42)

    def draw_branch(axes, branch, stable_colour, width=1.3, label=None):
        forcing = np.array(branch["forcing"])
        value = np.array(branch["thickness_min"])
        stable = np.array(branch["stable"], bool)
        drawn = False
        edges = np.where(np.diff(stable.astype(int)) != 0)[0] + 1
        for piece in np.split(np.arange(forcing.size), edges):
            if piece.size < 2:
                continue
            is_stable = bool(stable[piece[0]])
            axes.plot(forcing[piece], value[piece],
                      color=stable_colour if is_stable
                      else style.REGIME["unstable"],
                      linewidth=width if is_stable else 0.9,
                      linestyle="-" if is_stable else "--",
                      label=label if (is_stable and not drawn) else None,
                      zorder=3 if is_stable else 2)
            drawn = drawn or is_stable

    # (a) one S shaped curve in full.
    axes = axes_grid[0, 0]
    branch = branches[widest]
    window = branch.get("hysteresis")
    if window is not None:
        axes.axvspan(window["forcing_reverse"], window["forcing_forward"],
                     color=style.REGIME["bistable"], zorder=0)
    draw_branch(axes, branch, style.REGIME["cold"])
    for fold in branch["folds"]:
        axes.plot([fold["forcing"]], [fold["thickness_min"]], marker="o",
                  markersize=4.2, markerfacecolor=style.INK["surface"],
                  markeredgecolor=style.REGIME["warm"], markeredgewidth=1.1,
                  zorder=6)
    axes.axvline(0.0, color=style.INK["muted"], linewidth=0.7, linestyle=":")
    axes.annotate("Present day", xy=(0.0, axes.get_ylim()[1]),
                  xytext=(3, -7), textcoords="offset points", fontsize=6,
                  color=style.INK["secondary"], va="top")
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Late summer ice thickness (m)")
    axes.legend(handles=[
        Line2D([], [], color=style.REGIME["cold"], linewidth=1.3,
               label="Stable"),
        Line2D([], [], color=style.REGIME["unstable"], linewidth=0.9,
               linestyle="--", label="Unstable"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor=style.INK["surface"],
               markeredgecolor=style.REGIME["warm"], label="Fold")],
        loc="upper right", fontsize=5.8)
    style.panel_label(axes, "a")

    # (b) the whole family, coloured by transition scale.
    axes = axes_grid[0, 1]
    import matplotlib as mpl
    norm = mpl.colors.Normalize(min(scales), max(scales))
    for scale, branch in zip(scales, branches):
        draw_branch(axes, branch, style.SEQUENTIAL(norm(scale)), width=1.0)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=style.SEQUENTIAL)
    maps.add_colourbar(figure, mappable, axes, "Albedo transition scale (m)")
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Late summer ice thickness (m)")
    # The branches all leave the perennial state within this interval, so a
    # wider view would be mostly the flat ice free branch.
    axes.set_xlim(-7.0, 7.0)
    style.panel_label(axes, "b")

    # (c) width of the bistable window against the transition scale.
    axes = axes_grid[1, 0]
    widths = [(branch.get("hysteresis") or {"width": 0.0})["width"]
              for branch in branches]
    jumps = [(branch.get("hysteresis") or {"jump": 0.0})["jump"]
             for branch in branches]
    axes.plot(scales, widths, color=style.CATEGORICAL[0], marker="o",
              markersize=3.2, label="Window width")
    axes.set_xlabel("Albedo transition scale (m)")
    axes.set_ylabel("Bistable window (W m$^{-2}$)")

    twin = axes.twinx()
    twin.plot(scales, jumps, color=style.CATEGORICAL[1], marker="s",
              markersize=3.2, linestyle="--", label="Thickness jump")
    twin.set_ylabel("Jump in thickness (m)", color=style.INK["primary"],
                    labelpad=2.0)
    twin.spines["top"].set_visible(False)
    twin.tick_params(labelsize=6.5)

    handles = [Line2D([], [], color=style.CATEGORICAL[0], marker="o",
                      markersize=3.2, label="Window width"),
               Line2D([], [], color=style.CATEGORICAL[1], marker="s",
                      markersize=3.2, linestyle="--", label="Thickness jump")]
    axes.legend(handles=handles, loc="upper left", fontsize=5.8)
    style.soften_grid(axes)
    style.panel_label(axes, "c")

    # (d) the regimes over forcing and the transition scale.
    axes = axes_grid[1, 1]
    plane = bifurcation.get("parameter_plane")
    if plane is not None:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        forcing = np.array(plane["forcing"])
        plane_scales = np.array(plane["albedo_scale"])
        perennial = np.array(plane["perennial_possible"], bool)
        bistable = np.array(plane["bistable"], bool)
        field = np.where(bistable, 1, np.where(perennial, 2, 0)).T
        palette = ListedColormap([style.REGIME["warm"],
                                  style.REGIME["bistable"],
                                  style.REGIME["cold"]])
        axes.pcolormesh(forcing, plane_scales, field, cmap=palette,
                        vmin=-0.5, vmax=2.5, shading="auto")
        axes.axvline(0.0, color=style.INK["primary"], linewidth=0.8,
                     linestyle=":")
        axes.legend(handles=[
            Patch(facecolor=style.REGIME["cold"], label="Perennial ice"),
            Patch(facecolor=style.REGIME["bistable"], label="Both states"),
            Patch(facecolor=style.REGIME["warm"], label="Seasonal ice")],
            loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
            fontsize=5.6, frameon=False, columnspacing=1.0,
            handlelength=1.2)
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Albedo transition scale (m)")
    style.panel_label(axes, "d")

    style.save(figure, FIGURE_DIR / "figure2_bifurcation")
    plt.close(figure)
    print("  figure 2 written")


# --------------------------------------------------------------------------- #
# Figure 3: the regime map
# --------------------------------------------------------------------------- #

def figure_regimes(bifurcation: dict) -> None:
    """How the threshold moves as the ocean delivers more heat."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    regimes = bifurcation.get("regime_map")
    if regimes is None:
        print("  no regime map stored, skipping figure 3")
        return

    forcing = np.array(regimes["forcing"])
    atlantic = np.array(regimes["atlantic_temperature"])
    perennial = np.array(regimes["perennial_possible"], bool)
    bistable = np.array(regimes["bistable"], bool)
    thickness = np.array(regimes["thickness_min_from_perennial"])

    #: Present-day inflow temperature above freezing, K.
    present = 2.40

    style.use_style()
    figure, axes_grid = plt.subplots(1, 3, figsize=(style.WIDTH_DOUBLE, 2.5))
    figure.subplots_adjust(wspace=0.40)

    # (a) the three regimes.
    axes = axes_grid[0]
    field = np.where(bistable, 1, np.where(perennial, 2, 0)).T
    palette = ListedColormap([style.REGIME["warm"], style.REGIME["bistable"],
                              style.REGIME["cold"]])
    axes.pcolormesh(forcing, atlantic, field, cmap=palette, vmin=-0.5,
                    vmax=2.5, shading="auto")
    axes.plot([0.0], [present], marker="*", markersize=9.0,
              markerfacecolor=style.INK["surface"],
              markeredgecolor=style.INK["primary"], markeredgewidth=0.9,
              zorder=5)
    axes.annotate("Present day", xy=(0.0, present), xytext=(6, -10),
                  textcoords="offset points", fontsize=6,
                  color=style.INK["primary"])
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Atlantic inflow temperature (K)")
    axes.legend(handles=[
        Patch(facecolor=style.REGIME["cold"], label="Perennial ice"),
        Patch(facecolor=style.REGIME["bistable"], label="Both states"),
        Patch(facecolor=style.REGIME["warm"], label="Seasonal ice"),
        Line2D([], [], marker="*", linestyle="none", markersize=6.0,
               markerfacecolor=style.INK["surface"],
               markeredgecolor=style.INK["primary"], label="Present day")],
        loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
        fontsize=5.4, frameon=False, columnspacing=1.0, handlelength=1.2)
    style.panel_label(axes, "a")

    # (b) where the threshold sits as the inflow warms.
    axes = axes_grid[1]
    threshold = np.full(atlantic.size, np.nan)
    for j in range(atlantic.size):
        holding = np.where(perennial[:, j])[0]
        if holding.size:
            threshold[j] = forcing[holding.max()]
    good = np.isfinite(threshold)
    # The threshold is only known to lie between two columns of the grid, so it
    # carries an uncertainty of half the grid spacing.  Drawing that explicitly
    # is honest about where the staircase in the underlying map comes from.
    half_step = 0.5 * float(np.diff(forcing).mean())
    axes.errorbar(atlantic[good], threshold[good], yerr=half_step,
                  color=style.CATEGORICAL[0], marker="o", markersize=3.0,
                  linestyle="none", elinewidth=0.7, capsize=1.4)

    if good.sum() > 2:
        slope, intercept = np.polyfit(atlantic[good], threshold[good], 1)
        axes.plot(atlantic[good], slope * atlantic[good] + intercept,
                  color=style.SOURCE["emulator"], linewidth=0.9,
                  linestyle="--")
        axes.annotate(f"{slope:.2f} W m$^{{-2}}$ per K",
                      xy=(0.05, 0.10), xycoords="axes fraction", fontsize=6,
                      color=style.INK["secondary"])
    axes.axhline(0.0, color=style.INK["muted"], linewidth=0.8, linestyle=":")
    axes.axvline(present, color=style.INK["muted"], linewidth=0.8,
                 linestyle=":")
    axes.set_xlabel("Atlantic inflow temperature (K)")
    axes.set_ylabel("Threshold forcing (W m$^{-2}$)")
    style.soften_grid(axes)
    style.panel_label(axes, "b")

    # (c) how thick the summer ice is across the plane.
    axes = axes_grid[2]
    shown = np.where(perennial, thickness, np.nan)
    image = axes.pcolormesh(forcing, atlantic, shown.T, cmap=style.SEQUENTIAL,
                            shading="auto")
    maps.add_colourbar(figure, image, axes, "Late summer thickness (m)")
    axes.plot([0.0], [present], marker="*", markersize=9.0,
              markerfacecolor=style.INK["surface"],
              markeredgecolor=style.INK["primary"], markeredgewidth=0.9,
              zorder=5)
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Atlantic inflow temperature (K)")
    style.panel_label(axes, "c")

    style.save(figure, FIGURE_DIR / "figure3_regimes")
    plt.close(figure)
    print("  figure 3 written")


# --------------------------------------------------------------------------- #
# Figure 4: the pace of the forcing
# --------------------------------------------------------------------------- #

def figure_rate(tipping: dict) -> None:
    """The outcome depends on how far the forcing goes, not how fast."""
    import matplotlib.pyplot as plt

    style.use_style()
    figure, axes_grid = plt.subplots(1, 3, figsize=(style.WIDTH_DOUBLE, 2.5))
    figure.subplots_adjust(wspace=0.38)

    threshold = tipping["threshold"]

    # (a) outcome against ramp duration, for several targets.
    axes = axes_grid[0]
    colours = style.categorical(len(tipping["ramps"]))
    markers = ("o", "s", "^")
    for entry, colour, marker in zip(tipping["ramps"], colours, markers):
        axes.plot(entry["duration"], entry["thickness_min"], color=colour,
                  marker=marker, markersize=3.0, linewidth=1.1,
                  label=f"{entry['fraction']:.0%} of threshold")
    axes.set_xscale("log")
    axes.set_xlabel("Duration of the forcing ramp (yr)")
    axes.set_ylabel("Late summer ice thickness (m)")
    axes.set_ylim(bottom=0.0)
    axes.legend(loc="lower right", fontsize=5.8, title="Target",
                title_fontsize=5.8)
    style.soften_grid(axes)
    style.panel_label(axes, "a")

    overshoot = tipping.get("overshoot")
    if overshoot is None:
        style.save(figure, FIGURE_DIR / "figureS4_rate")
        plt.close(figure)
        print("  figure S4 written")
        return

    peaks = np.array(overshoot["peak"])
    forcing = np.array(overshoot["forcing"])
    thickness = np.array(overshoot["thickness"])
    times = np.array(overshoot["time"])
    recovered = np.array(overshoot["recovered"], bool)

    # (b) the trajectory in the plane of forcing and thickness.
    axes = axes_grid[1]
    shades = style.categorical(min(peaks.size, 6))
    for index in range(peaks.size):
        axes.plot(forcing[:, index], thickness[:, index],
                  color=shades[index % len(shades)], linewidth=1.0,
                  linestyle="-" if recovered[index] else "--")
    axes.axvline(threshold, color=style.REGIME["warm"], linewidth=0.9,
                 linestyle=":")
    axes.annotate("Threshold", xy=(threshold, axes.get_ylim()[1]),
                  xytext=(3, -6), textcoords="offset points", fontsize=6,
                  color=style.INK["secondary"], va="top")
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Ice thickness (m)")
    style.panel_label(axes, "b")

    # (c) the same runs against time.
    axes = axes_grid[2]
    for index in range(peaks.size):
        axes.plot(times, thickness[:, index],
                  color=shades[index % len(shades)], linewidth=1.0,
                  linestyle="-" if recovered[index] else "--",
                  label=f"{peaks[index] / threshold:.0%}")
    boundary = (overshoot["rise_years"] + overshoot["hold_years"]
                + overshoot["fall_years"])
    axes.axvline(boundary, color=style.INK["muted"], linewidth=0.8,
                 linestyle=":")
    axes.annotate("Forcing restored", xy=(boundary, axes.get_ylim()[1]),
                  xytext=(3, -6), textcoords="offset points", fontsize=6,
                  color=style.INK["secondary"], va="top")
    axes.set_xlabel("Year")
    axes.set_ylabel("Ice thickness (m)")
    axes.legend(loc="upper right", fontsize=5.5, title="Peak forcing",
                title_fontsize=5.5, ncol=2)
    style.soften_grid(axes)
    style.panel_label(axes, "c")

    style.save(figure, FIGURE_DIR / "figureS4_rate")
    plt.close(figure)
    print("  figure S4 written")


# --------------------------------------------------------------------------- #
# Figure 4: where the ice goes
# --------------------------------------------------------------------------- #

def figure_retreat(retreat: dict) -> None:
    """Where the ice is lost, followed in area and in mass together."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from arcticgnn.spatial.grid import build_grid

    grid = build_grid(retreat.get("spacing_km", 100.0))
    states = retreat["states"]
    forcings = np.array(retreat["forcings"])
    present = retreat["forcing_present"]

    style.use_style()
    figure = plt.figure(figsize=(style.WIDTH_DOUBLE, 5.4))
    # The lower row carries two colour bars, so its columns need more room than
    # the map row above.
    spec = figure.add_gridspec(2, 3, hspace=0.34, wspace=0.46,
                               height_ratios=[1.30, 1.0])

    # (a) to (c): September thickness at three points along the sequence.
    chosen = [0, int(np.argmin(np.abs(forcings))), len(states) - 2]
    letters = ("a", "b", "c")
    # A fixed ceiling well below the thickest ridged ice, so that the thin
    # cover which actually retreats is resolved instead of being compressed
    # into the pale end of the scale by a few thick cells.
    limit = 2.5
    reach = float(np.abs(grid.x).max()) + grid.spacing_km

    image = None
    for position, index, letter in zip(range(3), chosen, letters):
        axes = figure.add_subplot(spec[0, position])
        field = np.array(states[index]["thickness_september"])
        image = maps.draw_field(axes, grid, field, cmap=style.SEQUENTIAL,
                                vmin=0.0, vmax=limit)
        maps.draw_graticule(axes, reach, label_longitudes=False)
        axes.annotate(f"{forcings[index]:+.1f} W m$^{{-2}}$",
                      xy=(0.5, -0.04), xycoords="axes fraction", fontsize=6.2,
                      ha="center", va="top", color=style.INK["primary"])
        style.panel_label(axes, letter, dx=-0.02, dy=1.04)

    if image is not None:
        bar = figure.colorbar(image, ax=figure.axes[:3], fraction=0.020,
                              pad=0.015)
        bar.set_label("September ice thickness (m)", fontsize=6.5)
        bar.ax.tick_params(labelsize=6, length=2, width=0.5)
        bar.outline.set_visible(False)

    # (d) area and mass along the sequence.
    axes = figure.add_subplot(spec[1, 0])
    extent = np.array([s["extent_min"] for s in states])
    volume = np.array([s["volume_min"] for s in states])
    axes.plot(forcings, extent / extent[0], color=style.CATEGORICAL[0],
              marker="o", markersize=3.0, label="Area")
    axes.plot(forcings, volume / volume[0], color=style.CATEGORICAL[1],
              marker="s", markersize=3.0, linestyle="--", label="Mass")
    axes.axvline(0.0, color=style.INK["muted"], linewidth=0.8, linestyle=":")
    axes.annotate("Current climate\nperiod", xy=(0.0, 1.0), xytext=(4, -2),
                  textcoords="offset points", fontsize=5.8,
                  color=style.INK["secondary"], va="top")
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Fraction of the preindustrial value")
    axes.legend(loc="lower left", fontsize=6)
    style.soften_grid(axes)
    style.panel_label(axes, "d")

    # (e) mass against area, which shows they do not decline together.
    axes = figure.add_subplot(spec[1, 1])
    points = axes.scatter(extent, volume, c=forcings, cmap=style.SEQUENTIAL,
                          s=22, zorder=4, edgecolor=style.INK["surface"],
                          linewidth=0.3)
    axes.plot(extent, volume, color=style.INK["muted"], linewidth=0.7,
              zorder=3)
    maps.add_colourbar(figure, points, axes, "Forcing (W m$^{-2}$)",
                       pad=0.03, fraction=0.05)
    axes.set_xlabel("September ice area (10$^6$ km$^2$)")
    axes.set_ylabel("September ice mass (10$^3$ km$^3$)")
    style.soften_grid(axes)
    style.panel_label(axes, "e")

    # (f) where the summer ice survives longest.
    axes = figure.add_subplot(spec[1, 2])
    survival = np.array(retreat["survival"])
    image = maps.draw_field(axes, grid, survival, cmap=style.SEQUENTIAL,
                            vmin=0.0, vmax=1.0)
    maps.draw_contour(axes, grid, survival, [0.75], style.SOURCE["emulator"],
                      linewidth=1.0)
    maps.add_colourbar(figure, image, axes, "Fraction of the sequence with ice",
                       pad=0.03, fraction=0.05)
    style.panel_label(axes, "f", dx=-0.02, dy=1.04)

    style.save(figure, FIGURE_DIR / "figure4_retreat")
    plt.close(figure)
    print("  figure S4 written")


# --------------------------------------------------------------------------- #
# Figure 5: emulators and connectivity
# --------------------------------------------------------------------------- #

def figure_two_arctics(basins: dict, bifurcation: dict, tipping: dict,
                       calibration: dict) -> None:
    """Two Arctics that agree on the present and disagree about the future."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    pair = basins["pair"]
    monostable = next(e for e in pair if e["label"] == "monostable")
    bistable = next(e for e in pair if e["label"] == "bistable")
    branches = bifurcation["branches"]

    style.use_style()
    figure = plt.figure(figsize=(style.WIDTH_DOUBLE, 5.6))
    spec = figure.add_gridspec(2, 3, hspace=0.42, wspace=0.34)

    palette = ListedColormap([style.REGIME["warm"], style.REGIME["cold"]])

    # (a) the two annual cycles that coexist at one forcing.
    axes = figure.add_subplot(spec[0, 0])
    coexisting = bifurcation.get("coexisting") or {}
    cycles = coexisting.get("cycles", {})
    labels = {"perennial": "Ice survives the summer",
              "seasonal": "Ice free every summer"}
    colours = {"perennial": style.REGIME["cold"], "seasonal": style.REGIME["warm"]}
    for regime in ("perennial", "seasonal"):
        cycle = cycles.get(regime)
        if cycle is None:
            continue
        month = (np.array(cycle["time"]) % 1.0) * 12.0 + 1.0
        thickness = np.array(cycle["thickness"])
        order = np.argsort(month)
        axes.plot(month[order], thickness[order], color=colours[regime],
                  linewidth=1.4, label=labels[regime])
    axes.set_xlabel("Month")
    axes.set_ylabel("Ice thickness (m)")
    axes.set_xlim(1.0, 13.0)
    axes.set_xticks([1, 4, 7, 10, 13])
    axes.set_ylim(bottom=0.0)
    if coexisting:
        axes.annotate(f"Both at {coexisting['forcing']:+.2f} W m$^{{-2}}$",
                      xy=(0.04, 0.94), xycoords="axes fraction", fontsize=6,
                      color=style.INK["secondary"], va="top")
    axes.legend(loc="lower left", fontsize=5.8)
    style.soften_grid(axes)
    style.panel_label(axes, "a")

    # (b) the plane of states, coloured by which of those cycles it leads to.
    axes = figure.add_subplot(spec[0, 1])
    thickness_axis = np.array(bistable["basins"]["thickness"])
    deep_axis = np.array(bistable["basins"]["deep_temperature"])
    perennial = np.array(bistable["basins"]["perennial"], bool)
    axes.pcolormesh(thickness_axis, deep_axis, perennial.T.astype(float),
                    cmap=palette, vmin=-0.5, vmax=1.5, shading="auto",
                    rasterized=True)
    for regime, marker in (("perennial", "o"), ("seasonal", "s")):
        cycle = bistable["cycles"][regime]
        axes.plot(cycle["thickness"], cycle["deep_temperature"],
                  color=style.INK["primary"], linewidth=1.1)
        axes.plot([cycle["thickness"][0]], [cycle["deep_temperature"][0]],
                  marker=marker, markersize=3.6,
                  markerfacecolor=style.INK["surface"],
                  markeredgecolor=style.INK["primary"], markeredgewidth=0.9,
                  zorder=6)
    axes.set_xlabel("Ice thickness (m)")
    axes.set_ylabel("Ocean heat below the halocline (K)")
    style.panel_label(axes, "b")

    figure.legend(handles=[
        Line2D([], [], color=style.REGIME["cold"], linewidth=5,
               label="Leads to perennial ice"),
        Line2D([], [], color=style.REGIME["warm"], linewidth=5,
               label="Leads to ice free summers"),
        Line2D([], [], color=style.INK["primary"], linewidth=1.1,
               label="The annual cycles themselves")],
        loc="upper center", bbox_to_anchor=(0.52, 0.505), ncol=3,
        frameon=False, fontsize=5.8)

    # (c) the seasonal cycles, which are what observations see.
    axes = figure.add_subplot(spec[0, 2])
    months = np.arange(1, 13)
    for entry, colour, marker in ((monostable, style.CATEGORICAL[0], "o"),
                                  (bistable, style.CATEGORICAL[1], "s")):
        cycle = entry["control_cycle"]
        month = (np.array(cycle["time"]) % 1.0) * 12.0 + 1.0
        thickness = np.array(cycle["thickness"])
        order = np.argsort(month)
        axes.plot(month[order], thickness[order], color=colour,
                  linewidth=1.2, marker=marker, markersize=2.4, markevery=12,
                  label=f"{entry['albedo_scale']:.2f} m")
    axes.set_xlim(1.0, 13.0)

    column = calibration["column"]
    axes.axhline(column["target_max"], color=style.SOURCE["observations"],
                 linewidth=0.9, linestyle="--")
    axes.axhline(column["target_min"], color=style.SOURCE["observations"],
                 linewidth=0.9, linestyle="--")
    axes.annotate("Observed range", xy=(1.0, column["target_max"]),
                  xytext=(2, 3), textcoords="offset points", fontsize=5.8,
                  color=style.INK["secondary"])
    axes.set_xlabel("Month")
    axes.set_ylabel("Ice thickness (m)")
    axes.set_xticks([1, 4, 7, 10])
    axes.legend(loc="lower left", fontsize=5.8, title="Albedo scale",
                title_fontsize=5.8)
    style.soften_grid(axes)
    style.panel_label(axes, "c")

    # (d) the two branches, which is where they part company.
    axes = figure.add_subplot(spec[1, 0])
    for entry, colour in ((monostable, style.CATEGORICAL[0]),
                          (bistable, style.CATEGORICAL[1])):
        branch = branches[entry["branch_index"]]
        forcing = np.array(branch["forcing"])
        value = np.array(branch["thickness_min"])
        stable = np.array(branch["stable"], bool)
        edges = np.where(np.diff(stable.astype(int)) != 0)[0] + 1
        for piece in np.split(np.arange(forcing.size), edges):
            if piece.size < 2:
                continue
            solid = bool(stable[piece[0]])
            axes.plot(forcing[piece], value[piece], color=colour,
                      linewidth=1.2 if solid else 0.8,
                      linestyle="-" if solid else "--")
        window = branch.get("hysteresis")
        if window is not None:
            axes.axvspan(window["forcing_reverse"], window["forcing_forward"],
                         color=style.REGIME["bistable"], zorder=0)
    axes.axvline(0.0, color=style.INK["muted"], linewidth=0.8, linestyle=":")
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Late summer ice thickness (m)")
    axes.set_xlim(-4.0, 4.0)
    style.panel_label(axes, "d")

    # (e) the same forcing history applied to both.
    axes = figure.add_subplot(spec[1, 1])
    overshoot = tipping.get("overshoot")
    if overshoot is not None:
        peaks = np.array(overshoot["peak"])
        times = np.array(overshoot["time"])
        thickness = np.array(overshoot["thickness"])
        recovered = np.array(overshoot["recovered"], bool)
        threshold = tipping["threshold"]
        shades = style.categorical(min(peaks.size, 6))
        # The seasonal cycle is drawn in panel c; here the annual minimum is
        # plotted instead, because nine centuries of raw trajectory overlap into
        # a solid band that hides the trend it is meant to show.
        samples = max(int(round(times.size / max(times[-1] - times[0], 1.0))), 1)
        usable = (thickness.shape[0] // samples) * samples
        yearly = thickness[:usable].reshape(-1, samples, peaks.size).min(axis=1)
        year = times[:usable].reshape(-1, samples).mean(axis=1)
        for index in range(peaks.size):
            axes.plot(year, yearly[:, index],
                      color=shades[index % len(shades)], linewidth=1.1,
                      linestyle="-" if recovered[index] else "--",
                      label=f"{peaks[index] / threshold:.0%}")
        boundary = (overshoot["rise_years"] + overshoot["hold_years"]
                    + overshoot["fall_years"])
        axes.axvline(boundary, color=style.INK["muted"], linewidth=0.8,
                     linestyle=":")
        axes.annotate("Forcing restored", xy=(boundary, axes.get_ylim()[1]),
                      xytext=(-4, -6), textcoords="offset points", fontsize=5.6,
                      color=style.INK["secondary"], va="top", ha="right")
        axes.legend(loc="center right", fontsize=5.2, ncol=1,
                    title="Peak forcing", title_fontsize=5.2,
                    framealpha=0.85, edgecolor="none")
    axes.set_xlabel("Year")
    axes.set_ylabel("Late summer ice thickness (m)")
    style.soften_grid(axes)
    style.panel_label(axes, "e")

    # (f) what separates them, and what would need measuring.
    axes = figure.add_subplot(spec[1, 2])
    scales = np.array(bifurcation["albedo_scale"])
    widths = np.array([(branch.get("hysteresis") or {"width": 0.0})["width"]
                       for branch in branches])
    axes.fill_between(scales, 0.0, widths, color=style.REGIME["cold"],
                      alpha=0.25)
    axes.plot(scales, widths, color=style.CATEGORICAL[0], marker="o",
              markersize=3.0)
    for entry, colour in ((monostable, style.CATEGORICAL[0]),
                          (bistable, style.CATEGORICAL[1])):
        axes.plot([entry["albedo_scale"]],
                  [widths[entry["branch_index"]]], marker="*",
                  markersize=9.0, markerfacecolor=style.INK["surface"],
                  markeredgecolor=colour, markeredgewidth=1.1, zorder=6)
    axes.annotate("Reversible", xy=(scales.min(), 0.0), xytext=(4, 6),
                  textcoords="offset points", fontsize=6,
                  color=style.INK["secondary"])
    axes.annotate("Irreversible", xy=(scales.max(), widths.max()),
                  xytext=(-6, -8), textcoords="offset points", fontsize=6,
                  color=style.INK["secondary"], ha="right", va="top")
    axes.set_xlabel("Albedo transition scale (m)")
    axes.set_ylabel("Bistable window (W m$^{-2}$)")
    style.soften_grid(axes)
    style.panel_label(axes, "f")

    style.save(figure, FIGURE_DIR / "figure5_two_arctics")
    plt.close(figure)
    print("  figure 5 written")


# --------------------------------------------------------------------------- #
# Figure 6: early warning
# --------------------------------------------------------------------------- #

def figure_warning(warning: dict) -> None:
    """Whether the approach to the threshold is visible in advance."""
    import matplotlib.pyplot as plt

    member = warning["members"][0]
    record = member["record"]
    indicators = member["indicators"]

    style.use_style()
    figure, axes_grid = plt.subplots(
        2, 2, figsize=(style.WIDTH_DOUBLE, 4.2))
    figure.subplots_adjust(hspace=0.36, wspace=0.28)

    minimum = np.array(record["annual_minimum"], float)
    years = np.arange(minimum.size)
    loss = record["year_of_loss"]
    analysed = indicators["analysed_years"]

    # (a) the record itself.
    axes = axes_grid[0, 0]
    axes.plot(years, minimum, color=style.CATEGORICAL[0], linewidth=0.8)
    if loss is not None:
        axes.axvline(loss, color=style.REGIME["warm"], linewidth=0.9,
                     linestyle="--")
        axes.annotate("Ice lost", xy=(loss, minimum.max()), xytext=(-30, -6),
                      textcoords="offset points", fontsize=6,
                      color=style.INK["secondary"])
    axes.axvspan(0, analysed, color=style.REGIME["bistable"], alpha=0.5,
                 zorder=0)
    axes.set_xlabel("Year of the ramp")
    axes.set_ylabel("Late summer thickness (m)")
    style.panel_label(axes, "a")

    # (b) to (d) the indicators.
    order = (("variance", "Variance (m$^2$)", "b"),
             ("autocorrelation", "Lag one autocorrelation", "c"),
             ("irreversibility", "Time irreversibility", "d"))
    positions = ((0, 1), (1, 0), (1, 1))

    for (name, ylabel, letter), position in zip(order, positions):
        axes = axes_grid[position]
        index = np.array(indicators["index"], float)
        values = np.array(indicators[name], float)
        axes.plot(index, values, color=style.CATEGORICAL[0], linewidth=1.0)

        good = np.isfinite(values)
        if good.sum() > 2:
            fit = np.polyfit(index[good], values[good], 1)
            axes.plot(index[good], np.polyval(fit, index[good]),
                      color=style.SOURCE["emulator"], linewidth=0.9,
                      linestyle="--")
        trend = indicators["trends"][name]
        axes.annotate(f"$\\tau$ = {trend['tau']:+.2f}, "
                      f"p = {trend['p_value']:.3f}",
                      xy=(0.03, 0.92), xycoords="axes fraction", fontsize=6,
                      color=style.INK["secondary"], va="top")
        axes.set_xlabel("Year of the ramp")
        axes.set_ylabel(ylabel)
        style.soften_grid(axes)
        style.panel_label(axes, letter)

    style.save(figure, FIGURE_DIR / "figure6_warning")
    plt.close(figure)
    print("  figure 6 written")


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default=None,
                        help="build one figure: present, bifurcation, "
                             "regimes, rate, emulators, warning")
    arguments = parser.parse_args()

    ensure_dirs()
    calibration = load("calibration.json")
    bifurcation = load("bifurcation.json")
    tipping = load("tipping.json")
    emulators = load("emulators.json")
    warning = load("warning.json")
    basins = load("basins.json")
    retreat = load("retreat.json")

    wanted = arguments.only
    if calibration and wanted in (None, "present"):
        figure_present_day(calibration)
    if bifurcation and wanted in (None, "bifurcation"):
        figure_bifurcation(bifurcation)
    if bifurcation and wanted in (None, "regimes"):
        figure_regimes(bifurcation)
    if retreat and wanted in (None, "retreat"):
        figure_retreat(retreat)
    if (basins and bifurcation and tipping and calibration
            and wanted in (None, "arctics")):
        figure_two_arctics(basins, bifurcation, tipping, calibration)
    if warning and wanted in (None, "warning"):
        figure_warning(warning)

    print(f"figures are in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
