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

def _height_note(axes, text: str) -> None:
    """Name the vertical axis of a three dimensional panel inside the panel.

    ``Axes3D`` places a z label outside its own bounding box, far enough to the
    left that it lands on whatever sits in the neighbouring cell of the grid.
    Writing the quantity and its units in axes coordinates keeps the label with
    the panel it belongs to.
    """
    axes.text2D(0.10, 1.02, text, transform=axes.transAxes, fontsize=5.6,
                va="top", ha="left", color=style.INK["secondary"])


def _panel_label_3d(axes, letter: str, dx: float = -0.06,
                    dy: float = 1.02) -> None:
    """Place a panel letter on a three dimensional panel.

    ``Axes3D.text`` takes a third coordinate, so the flat helper in the style
    module cannot be used; ``text2D`` is the equivalent that writes in axes
    coordinates.
    """
    axes.text2D(dx, dy, letter, transform=axes.transAxes, fontsize=8.5,
                fontweight="bold", va="top", ha="left",
                color=style.INK["primary"])


def _style_3d(axes) -> None:
    """Strip a three dimensional panel back to the conventions of the study.

    The default Axes3D arrives with grey panes, a heavy grid and tick labels set
    at body size, none of which survive reduction to a third of a page width.
    The panes are washed out so the surface carries the panel, the grid is left
    in at a recessive weight because a surface without one is hard to read
    heights from, and the labels are brought down to the size used on the flat
    panels beside it.
    """
    for pane_axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        pane_axis.pane.set_facecolor(style.INK["surface"])
        pane_axis.pane.set_edgecolor(style.INK["grid"])
        pane_axis.pane.set_alpha(1.0)
        pane_axis._axinfo["grid"]["color"] = style.INK["grid"]
        pane_axis._axinfo["grid"]["linewidth"] = 0.3
    axes.tick_params(labelsize=5.4, pad=-1.5)
    axes.xaxis.labelpad = -3.0
    axes.yaxis.labelpad = -3.0


def figure_bifurcation(bifurcation: dict) -> None:
    """Branches of annual cycles, their folds, and the dependence on albedo."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers "3d")

    branches = bifurcation["branches"]
    scales = bifurcation["albedo_scale"]

    # The example panel uses the widest window found, since that is where the
    # structure is legible; the panel beside it shows the whole family.
    widest = max(range(len(branches)),
                 key=lambda i: (branches[i].get("hysteresis")
                                or {"width": -1.0})["width"])

    style.use_style()
    figure = plt.figure(figsize=(style.WIDTH_DOUBLE, 5.8))
    spec = figure.add_gridspec(2, 3, hspace=0.34, wspace=0.40,
                               left=0.075, right=0.975, top=0.94, bottom=0.11)

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

    # (a) one S shaped curve in full, with the fold itself enlarged inside it.
    axes = figure.add_subplot(spec[0, 0])
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
    axes.set_xlim(-7.5, 7.5)
    axes.legend(handles=[
        Line2D([], [], color=style.REGIME["cold"], linewidth=1.3,
               label="Stable"),
        Line2D([], [], color=style.REGIME["unstable"], linewidth=0.9,
               linestyle="--", label="Unstable"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor=style.INK["surface"],
               markeredgecolor=style.REGIME["warm"], label="Fold")],
        loc="lower left", fontsize=5.8)

    # The sub-panel carries the slow variable along the same branch.  Bistability
    # rests on the heat held below the halocline, and its rise across the window
    # is the part of the mechanism a thickness curve cannot show.
    sub = axes.inset_axes([0.55, 0.48, 0.41, 0.35])
    if window is not None:
        sub.axvspan(window["forcing_reverse"], window["forcing_forward"],
                    color=style.REGIME["bistable"], zorder=0)
    forcing = np.array(branch["forcing"])
    deep = np.array(branch["deep_temperature"])
    stable = np.array(branch["stable"], bool)
    sub.plot(forcing[stable], deep[stable], color=style.CATEGORICAL[2],
             linewidth=0.0, marker=".", markersize=1.1)
    sub.plot(forcing[~stable], deep[~stable], color=style.REGIME["unstable"],
             linewidth=0.0, marker=".", markersize=1.1)
    sub.set_xlim(-7.5, 7.5)
    sub.set_xlabel("forcing", fontsize=5.0, labelpad=0.8)
    sub.set_ylabel("deep T (K)", fontsize=5.0, labelpad=0.8)
    sub.tick_params(labelsize=4.8, length=1.8, pad=1.2)
    sub.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub.yaxis.set_major_locator(plt.MaxNLocator(3))
    for spine in sub.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(style.INK["muted"])
    style.panel_label(axes, "a")

    # (b) the whole family, coloured by transition scale, with the multiplier
    #     that decides stability shown inside.
    axes = figure.add_subplot(spec[0, 1])
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
    axes.set_xlim(-7.5, 7.5)

    sub = axes.inset_axes([0.56, 0.58, 0.40, 0.34])
    for scale, branch in zip(scales, branches):
        sub.plot(branch["forcing"], np.abs(branch["multiplier"]),
                 color=style.SEQUENTIAL(norm(scale)), linewidth=0.55)
    sub.axhline(1.0, color=style.REGIME["warm"], linewidth=0.7, linestyle="--")
    sub.set_xlim(-7.5, 7.5)
    sub.set_ylim(0.6, 1.4)
    sub.set_xlabel("forcing", fontsize=5.0, labelpad=0.8)
    sub.set_ylabel("|multiplier|", fontsize=5.0, labelpad=0.8)
    sub.tick_params(labelsize=4.8, length=1.8, pad=1.2)
    sub.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub.set_title("leading multiplier", fontsize=5.0,
                  color=style.INK["secondary"], pad=1.5)
    for spine in sub.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(style.INK["muted"])
    style.panel_label(axes, "b")
    # (c) the same family as a surface, which is where the fold reads as an
    #     edge running across the plane instead of a point on a curve.
    plane = bifurcation.get("parameter_plane")
    axes = figure.add_subplot(spec[0, 2], projection="3d")
    if plane is not None:
        plane_forcing = np.array(plane["forcing"])
        plane_scales = np.array(plane["albedo_scale"])
        height = np.array(plane["thickness_min_from_perennial"], float)
        height = np.where(np.isfinite(height), height, 0.0)
        mesh_f, mesh_s = np.meshgrid(plane_forcing, plane_scales,
                                     indexing="ij")
        floor = -0.45 * float(np.nanmax(height))
        axes.plot_surface(mesh_f, mesh_s, height, cmap=style.SEQUENTIAL,
                          rstride=1, cstride=1, linewidth=0.12,
                          edgecolors=style.INK["surface"], antialiased=True,
                          shade=True, alpha=0.97)
        axes.contourf(mesh_f, mesh_s, height, levels=12, zdir="z",
                      offset=floor, cmap=style.SEQUENTIAL, alpha=0.80)
        axes.contour(mesh_f, mesh_s, height, levels=[0.05], zdir="z",
                     offset=floor, colors=[style.REGIME["warm"]],
                     linewidths=1.0)
        axes.set_zlim(floor, float(np.nanmax(height)) * 1.02)
        axes.set_xlabel("Forcing (W m$^{-2}$)", fontsize=6.0)
        axes.set_ylabel("Transition scale (m)", fontsize=6.0)
        _height_note(axes, "height: late summer thickness (m)")
        axes.view_init(elev=27.0, azim=-128.0)
        _style_3d(axes)
    _panel_label_3d(axes, "c")

    # (d) width of the bistable window against the transition scale.
    axes = figure.add_subplot(spec[1, 0])
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

    # The same two quantities against one another: a window and a jump are two
    # readings of the same fold, and the sub-panel shows they move together.
    sub = axes.inset_axes([0.13, 0.40, 0.33, 0.30])
    sub.plot(widths, jumps, color=style.CATEGORICAL[2], marker="o",
             markersize=2.4, linewidth=0.8)
    sub.set_xlabel("window", fontsize=5.0, labelpad=0.8)
    sub.set_ylabel("jump", fontsize=5.0, labelpad=0.8)
    sub.tick_params(labelsize=4.8, length=1.8, pad=1.2)
    sub.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub.yaxis.set_major_locator(plt.MaxNLocator(3))
    for spine in sub.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(style.INK["muted"])
    style.panel_label(axes, "d")

    # (e) the regimes over forcing and the transition scale.
    axes = figure.add_subplot(spec[1, 1])
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
        # Colour alone separates the three regimes, so the band that carries the
        # result is hatched as well and the boundaries are drawn as lines.
        axes.contourf(forcing, plane_scales, np.where(bistable, 1.0, 0.0).T,
                      levels=[0.5, 1.5], colors="none",
                      hatches=["///"], zorder=2)
        axes.contour(forcing, plane_scales,
                     np.where(perennial, 1.0, 0.0).T, levels=[0.5],
                     colors=[style.INK["primary"]], linewidths=0.7, zorder=3)
        axes.axvline(0.0, color=style.INK["primary"], linewidth=0.8,
                     linestyle=":")
        axes.legend(handles=[
            Patch(facecolor=style.REGIME["cold"], label="Perennial ice"),
            Patch(facecolor=style.REGIME["bistable"], hatch="///",
                  label="Both states"),
            Patch(facecolor=style.REGIME["warm"], label="Seasonal ice")],
            loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3,
            fontsize=5.6, frameon=False, columnspacing=1.0,
            handlelength=1.2)
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Albedo transition scale (m)")
    style.panel_label(axes, "e")

    # (f) the two annual cycles that coexist at one forcing, with the orbits
    #     they trace in the plane of the two state variables inside.
    axes = figure.add_subplot(spec[1, 2])
    coexisting = bifurcation.get("coexisting")
    if coexisting is not None:
        cycles = coexisting["cycles"]
        pairs = (("perennial", style.REGIME["cold"], "Perennial cycle"),
                 ("seasonal", style.REGIME["warm"], "Seasonal cycle"))
        for name, colour, label in pairs:
            cycle = cycles.get(name)
            if cycle is None:
                continue
            months = np.array(cycle["time"]) * 12.0
            axes.plot(months, cycle["thickness"], color=colour, linewidth=1.3,
                      label=label, zorder=3)
            axes.fill_between(months, 0.0, cycle["thickness"], color=colour,
                              alpha=0.16, linewidth=0.0, zorder=1)
        axes.set_xlim(0.0, 12.0)
        axes.set_xticks([0, 3, 6, 9, 12])
        axes.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Jan"])
        axes.set_xlabel("Month of the year")
        axes.set_ylabel("Ice thickness (m)")
        axes.legend(loc="upper right", fontsize=5.8)
        axes.annotate(f"forcing {coexisting['forcing']:+.2f} W m$^{{-2}}$",
                      xy=(0.04, 0.06), xycoords="axes fraction", fontsize=5.6,
                      color=style.INK["secondary"])
        style.soften_grid(axes)

        sub = axes.inset_axes([0.46, 0.60, 0.46, 0.30])
        for name, colour, _ in pairs:
            cycle = cycles.get(name)
            if cycle is None:
                continue
            sub.plot(np.array(cycle["time"]) * 12.0,
                     np.array(cycle["open_fraction"]) * 100.0,
                     color=colour, linewidth=0.8)
        sub.set_xlim(0.0, 12.0)
        sub.set_xticks([0, 6, 12])
        sub.set_xticklabels(["Jan", "Jul", "Jan"])
        sub.set_ylabel("open water (%)", fontsize=5.0, labelpad=0.8)
        sub.tick_params(labelsize=4.8, length=1.8, pad=1.2)
        sub.yaxis.set_major_locator(plt.MaxNLocator(3))
        for spine in sub.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color(style.INK["muted"])
    style.panel_label(axes, "f")

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
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers "3d")

    regimes = bifurcation.get("regime_map")
    if regimes is None:
        print("  no regime map stored, skipping figure 3")
        return

    forcing = np.array(regimes["forcing"])
    atlantic = np.array(regimes["atlantic_temperature"])
    perennial = np.array(regimes["perennial_possible"], bool)
    bistable = np.array(regimes["bistable"], bool)
    thickness = np.array(regimes["thickness_min_from_perennial"])
    winter = np.array(regimes.get("thickness_max_from_perennial", thickness))

    #: Present-day inflow temperature above freezing, K.
    present = 2.40

    def mark_present(axes):
        axes.plot([0.0], [present], marker="*", markersize=9.0,
                  markerfacecolor=style.INK["surface"],
                  markeredgecolor=style.INK["primary"], markeredgewidth=0.9,
                  zorder=6)

    style.use_style()
    figure = plt.figure(figsize=(style.WIDTH_DOUBLE, 5.4))
    spec = figure.add_gridspec(2, 3, hspace=0.50, wspace=0.42,
                               left=0.075, right=0.965, top=0.94, bottom=0.14)

    # (a) the three regimes.
    axes = figure.add_subplot(spec[0, 0])
    field = np.where(bistable, 1, np.where(perennial, 2, 0)).T
    palette = ListedColormap([style.REGIME["warm"], style.REGIME["bistable"],
                              style.REGIME["cold"]])
    axes.pcolormesh(forcing, atlantic, field, cmap=palette, vmin=-0.5,
                    vmax=2.5, shading="auto")
    # The band where both states exist is hatched as well as coloured, so the
    # panel survives reproduction without colour.
    axes.contourf(forcing, atlantic, np.where(bistable, 1.0, 0.0).T,
                  levels=[0.5, 1.5], colors="none", hatches=["///"], zorder=2)
    axes.contour(forcing, atlantic, np.where(perennial, 1.0, 0.0).T,
                 levels=[0.5], colors=[style.INK["primary"]], linewidths=0.7,
                 zorder=3)
    mark_present(axes)
    axes.annotate("Present day", xy=(0.0, present), xytext=(6, -10),
                  textcoords="offset points", fontsize=6,
                  color=style.INK["primary"])
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Atlantic inflow temperature (K)")
    axes.legend(handles=[
        Patch(facecolor=style.REGIME["cold"], label="Perennial ice"),
        Patch(facecolor=style.REGIME["bistable"], hatch="///",
              label="Both states"),
        Patch(facecolor=style.REGIME["warm"], label="Seasonal ice"),
        Line2D([], [], marker="*", linestyle="none", markersize=6.0,
               markerfacecolor=style.INK["surface"],
               markeredgecolor=style.INK["primary"], label="Present day")],
        loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
        fontsize=5.4, frameon=False, columnspacing=1.0, handlelength=1.2)
    style.panel_label(axes, "a")

    # (b) where the threshold sits as the inflow warms.
    axes = figure.add_subplot(spec[0, 1])
    threshold = np.full(atlantic.size, np.nan)
    holding_fraction = perennial.mean(axis=0)
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

    # The threshold says where the fold sits; the share of the plane that still
    # holds ice through the summer says how much of the room has gone.
    sub = axes.inset_axes([0.55, 0.60, 0.40, 0.32])
    sub.plot(atlantic, holding_fraction * 100.0, color=style.CATEGORICAL[3],
             linewidth=0.9, marker="o", markersize=1.8)
    sub.set_xlabel("inflow T (K)", fontsize=5.0, labelpad=0.8)
    sub.set_ylabel("perennial (%)", fontsize=5.0, labelpad=0.8)
    sub.tick_params(labelsize=4.8, length=1.8, pad=1.2)
    sub.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub.yaxis.set_major_locator(plt.MaxNLocator(3))
    for spine in sub.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(style.INK["muted"])
    style.panel_label(axes, "b")

    # (c) the same plane as a surface, where the threshold is the cliff edge.
    axes = figure.add_subplot(spec[0, 2], projection="3d")
    # The surface only has height where perennial ice exists, so the view is cut
    # back to that part of the plane; drawn over the full range the sheet would
    # occupy a tenth of the panel.
    keep = forcing <= 2.5
    cut_forcing = forcing[keep]
    height = np.where(perennial[keep], thickness[keep], np.nan)
    solid = np.where(perennial[keep], thickness[keep], 0.0)
    mesh_f, mesh_a = np.meshgrid(cut_forcing, atlantic, indexing="ij")
    top = float(np.nanmax(height))
    floor = -0.55 * top
    axes.plot_surface(mesh_f, mesh_a, height, cmap=style.SEQUENTIAL,
                      rstride=1, cstride=1, linewidth=0.15,
                      edgecolors=style.INK["grid"], antialiased=True,
                      shade=False, vmin=0.0, vmax=top)
    axes.contourf(mesh_f, mesh_a, solid, levels=12, zdir="z", offset=floor,
                  cmap=style.SEQUENTIAL, alpha=0.85)
    axes.contour(mesh_f, mesh_a, np.where(perennial[keep], 1.0, 0.0),
                 levels=[0.5], zdir="z", offset=floor,
                 colors=[style.REGIME["warm"]], linewidths=1.0)
    axes.set_zlim(floor, top * 1.02)
    axes.set_xlabel("Forcing (W m$^{-2}$)", fontsize=6.0)
    axes.set_ylabel("Inflow T (K)", fontsize=6.0)
    _height_note(axes, "height: late summer thickness (m)")
    axes.view_init(elev=27.0, azim=-128.0)
    _style_3d(axes)
    _panel_label_3d(axes, "c")

    # (d) how thick the summer ice is across the plane.
    axes = figure.add_subplot(spec[1, 0])
    shown = np.where(perennial, thickness, np.nan)
    levels = np.linspace(0.0, float(np.nanmax(shown)), 13)
    image = axes.contourf(forcing, atlantic, shown.T, levels=levels,
                          cmap=style.SEQUENTIAL, extend="neither")
    lines = axes.contour(forcing, atlantic, shown.T, levels=levels[::3],
                         colors=[style.INK["secondary"]], linewidths=0.4)
    axes.clabel(lines, fmt="%.1f", fontsize=4.6, inline=True)
    maps.add_colourbar(figure, image, axes, "Late summer thickness (m)")
    mark_present(axes)
    axes.set_xlim(float(forcing.min()), 2.5)
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Atlantic inflow temperature (K)")
    style.panel_label(axes, "d")

    # (e) the winter cover over the same plane, which thins later and from a
    #     greater starting thickness than the summer cover.
    axes = figure.add_subplot(spec[1, 1])
    shown_winter = np.where(perennial, winter, np.nan)
    levels = np.linspace(0.0, float(np.nanmax(shown_winter)), 13)
    image = axes.contourf(forcing, atlantic, shown_winter.T, levels=levels,
                          cmap=style.SEQUENTIAL, extend="neither")
    lines = axes.contour(forcing, atlantic, shown_winter.T, levels=levels[::3],
                         colors=[style.INK["secondary"]], linewidths=0.4)
    axes.clabel(lines, fmt="%.1f", fontsize=4.6, inline=True)
    maps.add_colourbar(figure, image, axes, "Winter ice thickness (m)")
    mark_present(axes)
    axes.set_xlim(float(forcing.min()), 2.5)
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Atlantic inflow temperature (K)")
    style.panel_label(axes, "e")

    # (f) cuts through the plane at four inflow temperatures, which is where the
    #     shift of the whole curve towards the present day is legible.
    axes = figure.add_subplot(spec[1, 2])
    picks = np.linspace(0, atlantic.size - 1, 4).astype(int)
    hues = style.categorical(len(picks))
    for colour, j in zip(hues, picks):
        profile = np.where(perennial[:, j], thickness[:, j], 0.0)
        axes.plot(forcing, profile, color=colour, linewidth=1.1,
                  label=f"{atlantic[j]:.1f} K")
        if np.isfinite(threshold[j]):
            axes.plot([threshold[j]], [0.0], marker="v", markersize=3.4,
                      color=colour, clip_on=False, zorder=5)
    axes.axvline(0.0, color=style.INK["muted"], linewidth=0.8, linestyle=":")
    axes.annotate("Present day", xy=(0.0, 0.0), xytext=(4, 14),
                  textcoords="offset points", fontsize=5.6,
                  color=style.INK["secondary"], va="bottom")
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Late summer ice thickness (m)")
    axes.legend(title="Inflow temperature", loc="upper right", fontsize=5.4,
                title_fontsize=5.4)
    style.soften_grid(axes)
    style.panel_label(axes, "f")

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

    # Annual minima, because the raw trajectory carries a seasonal swing of more
    # than two metres and centuries of it overlap into a solid band.
    samples = max(int(round(times.size / max(times[-1] - times[0], 1.0))), 1)
    usable = (thickness.shape[0] // samples) * samples
    yearly = thickness[:usable].reshape(-1, samples, peaks.size).min(axis=1)
    yearly_forcing = forcing[:usable].reshape(-1, samples, peaks.size).mean(axis=1)
    year = times[:usable].reshape(-1, samples).mean(axis=1)

    # (b) the trajectory in the plane of forcing and thickness.
    axes = axes_grid[1]
    shades = style.categorical(min(peaks.size, 6))
    for index in range(peaks.size):
        axes.plot(yearly_forcing[:, index], yearly[:, index],
                  color=shades[index % len(shades)], linewidth=1.1,
                  linestyle="-" if recovered[index] else "--")
    axes.axvline(threshold, color=style.REGIME["warm"], linewidth=0.9,
                 linestyle=":")
    axes.annotate("Threshold", xy=(threshold, axes.get_ylim()[1]),
                  xytext=(3, -6), textcoords="offset points", fontsize=6,
                  color=style.INK["secondary"], va="top")
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Late summer ice thickness (m)")
    style.panel_label(axes, "b")

    # (c) the same runs against time.
    axes = axes_grid[2]
    for index in range(peaks.size):
        axes.plot(year, yearly[:, index],
                  color=shades[index % len(shades)], linewidth=1.1,
                  linestyle="-" if recovered[index] else "--",
                  label=f"{peaks[index] / threshold:.0%}")
    boundary = (overshoot["rise_years"] + overshoot["hold_years"]
                + overshoot["fall_years"])
    axes.axvline(boundary, color=style.INK["muted"], linewidth=0.8,
                 linestyle=":")
    # Below the axis, clear of the legend in the upper corner.
    axes.annotate("Forcing restored", xy=(boundary, 0.0),
                  xytext=(4, 10), textcoords="offset points", fontsize=5.8,
                  color=style.INK["secondary"], va="bottom")
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
# The warming already applied
# --------------------------------------------------------------------------- #

def figure_budget(budget: dict) -> None:
    """How much of the warming that ends perennial ice has already happened."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    entries = budget["entries"]
    applied = budget["forcing_present"]
    low, high = budget["forcing_present_range"]
    carrying = [e for e in entries if e.get("fraction_applied") is not None]

    style.use_style()
    figure = plt.figure(figsize=(style.WIDTH_DOUBLE, 6.8))
    # The upper panel carries its legend underneath, so it needs the extra gap.
    spec = figure.add_gridspec(3, 2, hspace=0.62, wspace=0.34,
                               height_ratios=[1.15, 1.0, 1.0])

    # (a) one bar per configuration, from preindustrial to its threshold.
    axes = figure.add_subplot(spec[0, :])
    positions = np.arange(len(carrying))
    for row, entry in zip(positions, carrying):
        total = entry["threshold_from_preindustrial"]
        # The whole distance from preindustrial conditions to the threshold.
        axes.add_patch(Rectangle((0.0, row - 0.34), total, 0.68,
                                 facecolor=style.REGIME["bistable"],
                                 edgecolor="none", zorder=2))
        # The part of it already applied.
        axes.add_patch(Rectangle((0.0, row - 0.34), applied, 0.68,
                                 facecolor=style.SOURCE["emulator"],
                                 edgecolor="none", zorder=3))
        axes.plot([total], [row], marker="|", markersize=11,
                  markeredgewidth=1.6, color=style.INK["primary"], zorder=5)
        axes.text(total + 0.12, row,
                  f"{100 * entry['fraction_applied']:.0f} %",
                  fontsize=6.2, va="center", ha="left",
                  color=style.INK["primary"])

    axes.axvline(applied, color=style.INK["primary"], linewidth=0.9,
                 linestyle="--", zorder=6)
    axes.axvspan(low, high, color=style.INK["primary"], alpha=0.08, zorder=1)
    # Below the lowest bar, where the axis is empty, so that it cannot collide
    # with the percentage labels that sit beside each threshold marker.
    axes.annotate("Warming applied by 1991 to 2020",
                  xy=(applied, len(carrying) - 0.42), xytext=(5, 0),
                  textcoords="offset points", fontsize=6,
                  color=style.INK["primary"], va="center")
    axes.set_yticks(positions)
    axes.set_yticklabels([f"{e['albedo_scale']:.2f}" for e in carrying])
    axes.set_ylabel("How gradually the ice surface\ndarkens as it thins (m)")
    axes.set_xlabel("Greenhouse forcing since before industrialisation "
                    "(W m$^{-2}$)")
    axes.set_xlim(0.0, max(e["threshold_from_preindustrial"]
                           for e in carrying) * 1.10)
    axes.set_ylim(-0.6, len(carrying) - 0.15)
    axes.legend(handles=[
        Line2D([], [], color=style.SOURCE["emulator"], linewidth=5,
               label="Warming already applied"),
        Line2D([], [], color=style.REGIME["bistable"], linewidth=5,
               label="Warming still remaining"),
        Line2D([], [], color=style.INK["primary"], linewidth=1.4,
               label="Ice stops surviving the summer")],
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
        fontsize=5.8, frameon=False, columnspacing=1.6, handlelength=1.6)
    style.panel_label(axes, "a", dx=-0.055)

    # (b) the fraction applied, with the forcing uncertainty carried through.
    axes = figure.add_subplot(spec[1, 0])
    scales = np.array([e["albedo_scale"] for e in carrying])
    fraction = 100 * np.array([e["fraction_applied"] for e in carrying])
    lower = 100 * np.array([e["fraction_low"] for e in carrying])
    upper = 100 * np.array([e["fraction_high"] for e in carrying])
    axes.fill_between(scales, lower, upper, color=style.SOURCE["emulator"],
                      alpha=0.20)
    axes.plot(scales, fraction, color=style.SOURCE["emulator"], marker="o",
              markersize=3.2)
    axes.axhline(100.0, color=style.INK["primary"], linewidth=0.8,
                 linestyle="--")
    axes.annotate("All of it applied: ice no longer\nsurvives the summer",
                  xy=(scales.min(), 100.0), xytext=(2, 3),
                  textcoords="offset points", fontsize=5.6,
                  color=style.INK["secondary"])
    axes.set_xlabel("Darkening scale (m)")
    axes.set_ylabel("Percentage of the warming\nalready applied")
    axes.set_ylim(0.0, 118.0)
    style.soften_grid(axes)

    # What is left, in the units policy is written in.
    remaining = np.array([e["remaining_kelvin"] for e in carrying])
    sub = axes.inset_axes([0.54, 0.12, 0.40, 0.30])
    sub.plot(scales, remaining, color=style.CATEGORICAL[3], linewidth=0.9,
             marker="o", markersize=1.8)
    sub.axhline(0.0, color=style.INK["muted"], linewidth=0.6, linestyle=":")
    sub.set_xlabel("darkening scale (m)", fontsize=5.0, labelpad=0.8)
    sub.set_ylabel("still to come (K)", fontsize=5.0, labelpad=0.8)
    sub.tick_params(labelsize=4.8, length=1.8, pad=1.2)
    sub.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub.yaxis.set_major_locator(plt.MaxNLocator(3))
    for spine in sub.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(style.INK["muted"])
    style.panel_label(axes, "b")

    # (c) thinning that has already occurred.
    axes = figure.add_subplot(spec[1, 1])
    every = np.array([e["albedo_scale"] for e in entries])
    before = np.array([e["thickness_preindustrial"] for e in entries])
    now = np.array([e["thickness_present"] for e in entries])
    axes.fill_between(every, now, before, color=style.CATEGORICAL[0],
                      alpha=0.20, label="Thickness already lost")
    axes.plot(every, before, color=style.CATEGORICAL[0], marker="o",
              markersize=2.8, label="Before industrialisation")
    axes.plot(every, now, color=style.SOURCE["emulator"], marker="s",
              markersize=2.8, linestyle="--", label="Today (1991 to 2020)")
    axes.set_xlabel("Darkening scale (m)")
    axes.set_ylabel("Mean ice thickness (m)")
    axes.legend(loc="upper left", fontsize=5.6)
    style.soften_grid(axes)
    style.panel_label(axes, "c")

    # (d) the seasonal range of each configuration, then and now.  A mean
    #     thickness can hold steady while the range beneath it collapses, so the
    #     two extremes are drawn as bands instead of a single line.
    axes = figure.add_subplot(spec[2, 0])
    before_max = np.array([v for v in budget["preindustrial"]["max"]], float)
    before_min = np.array([v for v in budget["preindustrial"]["min"]], float)
    now_max = np.array([v for v in budget["present"]["max"]], float)
    now_min = np.array([v for v in budget["present"]["min"]], float)
    axes.fill_between(every, before_min, before_max,
                      color=style.CATEGORICAL[0], alpha=0.22,
                      label="Before industrialisation")
    axes.fill_between(every, now_min, now_max, color=style.SOURCE["emulator"],
                      alpha=0.30, label="Today (1991 to 2020)")
    axes.plot(every, before_max, color=style.CATEGORICAL[0], linewidth=0.9)
    axes.plot(every, before_min, color=style.CATEGORICAL[0], linewidth=0.9,
              linestyle="--")
    axes.plot(every, now_max, color=style.SOURCE["emulator"], linewidth=0.9)
    axes.plot(every, now_min, color=style.SOURCE["emulator"], linewidth=0.9,
              linestyle="--")
    axes.annotate("winter", xy=(every[-1], before_max[-1]), xytext=(-2, 4),
                  textcoords="offset points", fontsize=5.4, ha="right",
                  color=style.INK["secondary"])
    axes.annotate("late summer", xy=(every[-1], now_min[-1]), xytext=(-2, -8),
                  textcoords="offset points", fontsize=5.4, ha="right",
                  color=style.INK["secondary"])
    axes.set_xlabel("Darkening scale (m)")
    axes.set_ylabel("Ice thickness (m)")
    axes.legend(loc="upper left", fontsize=5.6)
    style.soften_grid(axes)
    style.panel_label(axes, "d")

    # (e) the thinning already recorded set against the distance still to run.
    #     A configuration that has thinned most is not the one with least room
    #     left, and the panel is where that can be checked.
    axes = figure.add_subplot(spec[2, 1])
    thinning = np.array([e["thinning"] for e in carrying])
    left = np.array([e["remaining"] for e in carrying])
    points = axes.scatter(thinning, left, c=scales, cmap=style.SEQUENTIAL,
                          s=26, zorder=4, edgecolor=style.INK["surface"],
                          linewidth=0.3)
    axes.plot(thinning, left, color=style.INK["muted"], linewidth=0.7,
              zorder=3)
    axes.axhline(0.0, color=style.INK["primary"], linewidth=0.8,
                 linestyle="--")
    axes.annotate("threshold already passed", xy=(0.04, 0.06),
                  xycoords="axes fraction", fontsize=5.4,
                  color=style.INK["secondary"])
    maps.add_colourbar(figure, points, axes, "Darkening scale (m)",
                       pad=0.03, fraction=0.05)
    axes.set_xlabel("Mean thickness already lost (m)")
    axes.set_ylabel("Forcing still to come (W m$^{-2}$)")
    style.soften_grid(axes)
    style.panel_label(axes, "e")

    style.save(figure, FIGURE_DIR / "figure_budget")
    plt.close(figure)
    print("  budget figure written")


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
    figure = plt.figure(figsize=(style.WIDTH_DOUBLE, 7.3))
    # The middle and lower rows carry colour bars, so their columns need more
    # room than the map row above.
    spec = figure.add_gridspec(3, 3, hspace=0.30, wspace=0.46,
                               height_ratios=[1.02, 1.0, 1.0])

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
                              pad=0.015, shrink=0.80)
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
    # Set along the rule it names, in the wedge under the curves that neither
    # the legend nor the sub-panel occupies.
    axes.annotate("Current climate period", xy=(0.0, 0.16), xytext=(-3, 0),
                  textcoords="offset points", fontsize=5.8, rotation=90,
                  color=style.INK["secondary"], va="bottom", ha="right")
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Fraction of the preindustrial value")
    axes.legend(loc="lower center", ncol=2, fontsize=6, columnspacing=1.2)
    style.soften_grid(axes)

    # Mass divided by area is the mean thickness of what is left, and it is the
    # single number that says the remaining ice is thinner as well as smaller.
    # It goes in the upper right, the one corner both curves leave empty.
    sub = axes.inset_axes([0.60, 0.62, 0.36, 0.30])
    sub.plot(forcings, volume / np.maximum(extent, 1e-9),
             color=style.CATEGORICAL[2], linewidth=0.9, marker="o",
             markersize=1.8)
    sub.axvline(0.0, color=style.INK["muted"], linewidth=0.6, linestyle=":")
    sub.set_xlabel("forcing", fontsize=5.0, labelpad=0.8)
    sub.set_ylabel("mean thickness (m)", fontsize=5.0, labelpad=0.8)
    sub.tick_params(labelsize=4.8, length=1.8, pad=1.2)
    sub.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub.yaxis.set_major_locator(plt.MaxNLocator(3))
    for spine in sub.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(style.INK["muted"])
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

    # The cells are gathered into latitude bands so that the retreat can be read
    # as one field over forcing, which no single map shows.
    latitude = np.array(retreat["latitude"])
    edges = np.linspace(float(latitude.min()), float(latitude.max()), 19)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bands = np.digitize(latitude, edges) - 1
    bands = np.clip(bands, 0, centres.size - 1)
    profile = np.zeros((forcings.size, centres.size))
    for index, state in enumerate(states):
        field = np.array(state["thickness_september"])
        for band in range(centres.size):
            inside = bands == band
            if inside.any():
                profile[index, band] = float(field[inside].mean())

    # (g) the same retreat as a field over forcing and latitude.
    axes = figure.add_subplot(spec[2, 0])
    levels = np.linspace(0.0, float(profile.max()), 13)
    image = axes.contourf(forcings, centres, profile.T, levels=levels,
                          cmap=style.SEQUENTIAL)
    lines = axes.contour(forcings, centres, profile.T, levels=levels[::4],
                         colors=[style.INK["secondary"]], linewidths=0.4)
    axes.clabel(lines, fmt="%.1f", fontsize=4.6, inline=True)
    axes.axvline(0.0, color=style.INK["surface"], linewidth=0.9,
                 linestyle=":")
    maps.add_colourbar(figure, image, axes, "September thickness (m)",
                       pad=0.09, fraction=0.05)
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Latitude (degrees north)")
    style.panel_label(axes, "g")

    # (h) the same field as a surface, where the retreat towards the pole is a
    #     slope and the last ice is the ridge that survives it.
    axes = figure.add_subplot(spec[2, 1], projection="3d")
    mesh_x, mesh_y = np.meshgrid(forcings, centres, indexing="ij")
    top = float(profile.max())
    floor = -0.55 * top
    axes.plot_surface(mesh_x, mesh_y, profile, cmap=style.SEQUENTIAL,
                      rstride=1, cstride=1, linewidth=0.15,
                      edgecolors=style.INK["grid"], antialiased=True,
                      shade=False, vmin=0.0, vmax=top)
    axes.contourf(mesh_x, mesh_y, profile, levels=12, zdir="z", offset=floor,
                  cmap=style.SEQUENTIAL, alpha=0.85)
    axes.set_zlim(floor, top * 1.02)
    axes.set_xlabel("Forcing (W m$^{-2}$)", fontsize=6.0)
    axes.set_ylabel("Latitude (deg N)", fontsize=6.0)
    _height_note(axes, "height: September thickness (m)")
    axes.view_init(elev=27.0, azim=-128.0)
    _style_3d(axes)
    _panel_label_3d(axes, "h")

    # (i) the two ends of the seasonal cycle, which leave at different times.
    axes = figure.add_subplot(spec[2, 2])
    extent_max = np.array([s["extent_max"] for s in states])
    volume_max = np.array([s["volume_max"] for s in states])
    axes.plot(forcings, extent_max / extent_max[0], color=style.CATEGORICAL[0],
              marker="o", markersize=2.6, label="Winter area")
    axes.plot(forcings, extent / extent[0], color=style.CATEGORICAL[0],
              marker="o", markersize=2.6, linestyle=":", label="September area")
    axes.plot(forcings, volume_max / volume_max[0], color=style.CATEGORICAL[1],
              marker="s", markersize=2.6, linestyle="--", label="Winter mass")
    axes.plot(forcings, volume / volume[0], color=style.CATEGORICAL[1],
              marker="s", markersize=2.6, linestyle="-.",
              label="September mass")
    axes.axvline(0.0, color=style.INK["muted"], linewidth=0.8, linestyle=":")
    axes.set_xlabel("Greenhouse forcing (W m$^{-2}$)")
    axes.set_ylabel("Fraction of the preindustrial value")
    axes.legend(loc="lower left", fontsize=5.2)
    style.soften_grid(axes)
    style.panel_label(axes, "i")

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

    style.save(figure, FIGURE_DIR / "figureS5_warning")
    plt.close(figure)
    print("  figure S5 written")


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
    budget = load("budget.json")

    wanted = arguments.only
    if calibration and wanted in (None, "present"):
        figure_present_day(calibration)
    if bifurcation and wanted in (None, "bifurcation"):
        figure_bifurcation(bifurcation)
    if bifurcation and wanted in (None, "regimes"):
        figure_regimes(bifurcation)
    if budget and wanted in (None, "budget"):
        figure_budget(budget)
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
