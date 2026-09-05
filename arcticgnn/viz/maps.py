"""Drawing fields on the polar grid.

The model grid is already an azimuthal projection centred on the pole, so a
field can be shown directly in projection coordinates without reprojecting
anything.  Coastlines are drawn from the same Natural Earth polygons that built
the land mask, so the outline and the mask agree exactly; taking them from
different sources leaves ice apparently sitting on land.
"""

from __future__ import annotations

import numpy as np

from . import style

#: Radius of the Earth, km.
EARTH_RADIUS_KM = 6371.0


def latlon_to_polar(latitude: np.ndarray, longitude: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the north polar azimuthal equidistant plane, km."""
    radius = np.radians(90.0 - np.asarray(latitude, float)) * EARTH_RADIUS_KM
    angle = np.radians(np.asarray(longitude, float))
    return radius * np.sin(angle), radius * np.cos(angle)


def draw_coastline(axes, max_radius_km: float, colour: str | None = None,
                   linewidth: float = 0.4) -> None:
    """Draw coastlines inside the domain."""
    colour = colour if colour is not None else style.INK["secondary"]
    try:
        import cartopy.io.shapereader as shapereader
    except ImportError:                                  # pragma: no cover
        return

    path = shapereader.natural_earth(resolution="50m", category="physical",
                                     name="coastline")
    for geometry in shapereader.Reader(path).geometries():
        lines = (geometry.geoms if hasattr(geometry, "geoms") else [geometry])
        for line in lines:
            longitude, latitude = np.asarray(line.coords).T
            if latitude.max() < 40.0:
                continue
            x, y = latlon_to_polar(latitude, longitude)
            inside = np.hypot(x, y) <= max_radius_km * 1.02
            if not inside.any():
                continue
            # Break the line wherever it leaves the domain, so that segments are
            # not joined straight across it.
            breaks = np.where(np.diff(inside.astype(int)) != 0)[0] + 1
            for piece in np.split(np.arange(x.size), breaks):
                if piece.size > 1 and inside[piece[0]]:
                    axes.plot(x[piece], y[piece], color=colour,
                              linewidth=linewidth, zorder=4,
                              solid_capstyle="round")


def draw_field(axes, grid, values: np.ndarray, cmap=None, vmin: float = None,
               vmax: float = None, norm=None, coastline: bool = True,
               land_colour: str = "#EDEAE4"):
    """Draw a per-cell field as a map, returning the image handle."""
    field = grid.to_map(np.asarray(values, float))
    reach = float(np.abs(grid.x).max()) + grid.spacing_km

    axes.set_facecolor(land_colour)
    image = axes.imshow(
        field.T, origin="lower",
        extent=(-reach, reach, -reach, reach),
        cmap=cmap if cmap is not None else style.SEQUENTIAL,
        vmin=vmin, vmax=vmax, norm=norm, interpolation="nearest", zorder=2)

    if coastline:
        draw_coastline(axes, reach)

    axes.set_xlim(-reach, reach)
    axes.set_ylim(-reach, reach)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    return image


def draw_contour(axes, grid, values: np.ndarray, levels, colour: str,
                 linewidth: float = 0.9, linestyle: str = "-",
                 outside: float | None = None):
    """Overlay a contour of a per-cell field.

    Parameters
    ----------
    outside
        Value assigned to land and to cells beyond the domain before
        contouring.  Leave it as the default, which masks them, unless there is
        a reason not to.  This matters for an ice edge: any finite fill places
        the contour level somewhere between the coastal ocean cells and the land
        beside them, so the line traces the coastline and the rim of the domain,
        and the real ice edge is lost among the loops.  Masked cells are skipped
        by the contouring routine, leaving only the edge in open water.
    """
    fill = outside if outside is not None else np.nan
    field = grid.to_map(np.asarray(values, float), fill=fill)
    reach = float(np.abs(grid.x).max()) + grid.spacing_km
    coordinate = np.linspace(-reach, reach, field.shape[0])
    mesh_x, mesh_y = np.meshgrid(coordinate, coordinate, indexing="ij")
    return axes.contour(mesh_x, mesh_y, field, levels=levels, colors=colour,
                        linewidths=linewidth, linestyles=linestyle, zorder=5)


def draw_graticule(axes, max_radius_km: float,
                   latitudes: tuple = (60.0, 70.0, 80.0),
                   longitudes: tuple = (-180.0, -120.0, -60.0, 0.0, 60.0, 120.0),
                   colour: str | None = None, linewidth: float = 0.35,
                   label_latitudes: bool = True,
                   label_longitudes: bool = True) -> None:
    """Draw parallels and meridians, with labels around the edge of the domain.

    Latitude circles are concentric in this projection and meridians are
    straight lines through the pole, so the graticule costs nothing to construct
    and gives the reader the geographic reference a polar map otherwise lacks.
    """
    colour = colour if colour is not None else style.INK["muted"]
    angle = np.linspace(0.0, 2.0 * np.pi, 361)

    for latitude in latitudes:
        radius = np.radians(90.0 - latitude) * EARTH_RADIUS_KM
        if radius > max_radius_km:
            continue
        axes.plot(radius * np.sin(angle), radius * np.cos(angle), color=colour,
                  linewidth=linewidth, linestyle=":", zorder=3)
        if label_latitudes:
            # Along the meridian towards the lower left, which is open ocean in
            # this domain and so free of coastline.
            axes.text(radius * np.sin(np.radians(225.0)),
                      radius * np.cos(np.radians(225.0)),
                      f"{latitude:.0f}\u00b0N", fontsize=4.8, color=colour,
                      ha="center", va="center", zorder=8,
                      bbox=dict(boxstyle="round,pad=0.10", fc=style.INK["surface"],
                                ec="none", alpha=0.75))

    for longitude in longitudes:
        direction = np.radians(longitude)
        axes.plot([0.0, max_radius_km * np.sin(direction)],
                  [0.0, max_radius_km * np.cos(direction)], color=colour,
                  linewidth=linewidth, linestyle=":", zorder=3)
        if label_longitudes:
            edge = 1.045 * max_radius_km
            axes.text(edge * np.sin(direction), edge * np.cos(direction),
                      f"{abs(longitude):.0f}\u00b0"
                      + ("" if longitude in (0.0, 180.0, -180.0)
                         else ("E" if longitude > 0 else "W")),
                      fontsize=4.8, color=colour, ha="center", va="center",
                      zorder=8)


def draw_ice_edge(axes, segments, max_radius_km: float, colour: str,
                  linewidth: float = 1.1, linestyle: str = "-",
                  label: str | None = None):
    """Overlay an observed ice edge given as latitude and longitude segments.

    The segments are projected with the same transformation as the model grid,
    so the observed edge and the modelled field are directly comparable.
    """
    handle = None
    for segment in segments:
        x, y = latlon_to_polar(segment[:, 0], segment[:, 1])
        inside = np.hypot(x, y) <= max_radius_km
        if not inside.any():
            continue
        # Split wherever the line leaves the domain so that pieces are not
        # joined straight across it.
        breaks = np.where(np.diff(inside.astype(int)) != 0)[0] + 1
        for piece in np.split(np.arange(x.size), breaks):
            if piece.size > 1 and inside[piece[0]]:
                lines = axes.plot(x[piece], y[piece], color=colour,
                                  linewidth=linewidth, linestyle=linestyle,
                                  zorder=7, solid_capstyle="round")
                handle = handle or lines[0]
    if handle is not None and label is not None:
        handle.set_label(label)
    return handle


def draw_vectors(axes, grid, u: np.ndarray, v: np.ndarray, step: int = 4,
                 colour: str | None = None, scale: float = None):
    """Overlay a thinned vector field."""
    colour = colour if colour is not None else style.INK["primary"]
    keep = slice(None, None, step)
    return axes.quiver(grid.x[keep], grid.y[keep], u[keep], v[keep],
                       color=colour, width=0.004, scale=scale, zorder=6,
                       alpha=0.8)


def add_colourbar(figure, image, axes, label: str, pad: float = 0.02,
                  fraction: float = 0.046, orientation: str = "vertical"):
    """Attach a colour bar with a label, sized to the axes it belongs to."""
    bar = figure.colorbar(image, ax=axes, fraction=fraction, pad=pad,
                          orientation=orientation)
    bar.set_label(label, fontsize=6.5, color=style.INK["primary"])
    bar.ax.tick_params(labelsize=6, length=2, width=0.5,
                       color=style.INK["secondary"])
    bar.outline.set_visible(False)
    return bar
