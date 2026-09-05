"""Pan-Arctic grid, land mask and graph construction.

The spatial model runs on an equal-area azimuthal grid centred on the North
Pole, so that every cell carries the same weight in an area average and no
cosine factors are needed when summing ice area.  Land is taken from the Natural
Earth coastline at 50 m resolution, which keeps the geography real: the Fram
Strait opening, the Canadian Arctic Archipelago and the Barents Sea shelf all
appear where they belong, and the ocean cells form a single connected basin.

The same cell layout serves two purposes.  It is the finite-volume mesh of the
spatial model, and it is the node set of the graph that the emulators are
trained on.  Because both use one geometry, an edge in the graph corresponds to
a physically meaningful neighbour relation and the length scales recovered from
the trained attention weights can be quoted in kilometres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DATA_DIR

#: Radius of the Earth, km.
EARTH_RADIUS_KM = 6371.0

#: Southern limit of the domain, degrees north.  The Arctic Ocean and its
#: marginal seas lie north of this.
DOMAIN_LATITUDE = 55.0


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #

@dataclass
class ArcticGrid:
    """An equal-area polar grid restricted to ocean cells.

    Attributes
    ----------
    x, y
        Cell centres in the azimuthal equidistant projection, km from the pole.
    latitude, longitude
        Cell centres in degrees.
    ocean
        Index of each retained cell in the full square grid, for reshaping back
        to a map.
    shape
        Shape of the full square grid.
    spacing_km
        Cell size, km.
    area_km2
        Area of one cell, km^2.
    """

    x: np.ndarray
    y: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    ocean: np.ndarray
    shape: tuple
    spacing_km: float
    area_km2: float
    #: Boolean over the full square grid, true where a cell is land inside the
    #: domain.  Needed to tell a coast from the open southern boundary: ice may
    #: leave the domain across the latter but must not vanish into the former.
    land: np.ndarray = None

    @property
    def n_cells(self) -> int:
        return int(self.x.size)

    def to_map(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Scatter per-cell values back onto the full square grid."""
        values = np.asarray(values, float)
        out = np.full(self.shape[0] * self.shape[1], fill, float)
        out[self.ocean] = values
        return out.reshape(self.shape)

    def distance_km(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Straight-line distance between pairs of cells, km."""
        return np.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j])


def _polar_to_latlon(x_km: np.ndarray, y_km: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Invert the north polar azimuthal equidistant projection.

    Distance from the pole maps linearly to colatitude, which is what makes the
    projection equidistant; the grid is treated as equal area because the cells
    are small compared with the radius of the Earth.
    """
    radius = np.hypot(x_km, y_km)
    colatitude = np.degrees(radius / EARTH_RADIUS_KM)
    latitude = 90.0 - colatitude
    longitude = np.degrees(np.arctan2(x_km, y_km))
    return latitude, longitude


def _land_mask(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Boolean mask of cells that fall on land.

    Uses the Natural Earth land polygons through Cartopy.  If those are not
    available the caller is told plainly, because silently substituting a
    circular ocean would change every length scale the study reports.
    """
    try:
        import cartopy.io.shapereader as shapereader
        from shapely.geometry import Point
        from shapely.prepared import prep
        from shapely.ops import unary_union
    except ImportError as exc:                      # pragma: no cover
        raise RuntimeError(
            "the land mask needs Cartopy and Shapely; install them rather than "
            "falling back to an idealised basin, which would invalidate the "
            "length scales reported from the graph") from exc

    path = shapereader.natural_earth(resolution="50m", category="physical",
                                     name="land")
    land = prep(unary_union(list(shapereader.Reader(path).geometries())))
    return np.array([land.contains(Point(lon, lat))
                     for lat, lon in zip(latitude, longitude)], bool)


def build_grid(spacing_km: float = 100.0,
               cache: bool = True) -> ArcticGrid:
    """Construct the ocean grid at the requested spacing.

    Parameters
    ----------
    spacing_km
        Cell size.  100 km resolves the main basins and straits while keeping
        the node count low enough to train graph networks on a laptop.
    cache
        Store the result under the data directory and reuse it, since building
        the land mask involves a point in polygon test per cell.
    """
    cache_path = (DATA_DIR /
                  f"arctic_grid_{int(spacing_km)}km"
                  f"_{int(DOMAIN_LATITUDE)}n.npz")
    if cache and cache_path.exists():
        stored = np.load(cache_path)
        return ArcticGrid(
            x=stored["x"], y=stored["y"], latitude=stored["latitude"],
            longitude=stored["longitude"], ocean=stored["ocean"],
            shape=tuple(stored["shape"]), spacing_km=float(stored["spacing_km"]),
            area_km2=float(stored["area_km2"]), land=stored["land"])

    reach_km = np.radians(90.0 - DOMAIN_LATITUDE) * EARTH_RADIUS_KM
    coordinate = np.arange(-reach_km, reach_km + spacing_km, spacing_km)
    grid_x, grid_y = np.meshgrid(coordinate, coordinate, indexing="ij")
    shape = grid_x.shape

    flat_x, flat_y = grid_x.ravel(), grid_y.ravel()
    latitude, longitude = _polar_to_latlon(flat_x, flat_y)

    inside = latitude >= DOMAIN_LATITUDE
    candidate = np.where(inside)[0]
    is_land = _land_mask(latitude[candidate], longitude[candidate])
    ocean = candidate[~is_land]

    land = np.zeros(flat_x.size, bool)
    land[candidate[is_land]] = True

    grid = ArcticGrid(
        x=flat_x[ocean], y=flat_y[ocean],
        latitude=latitude[ocean], longitude=longitude[ocean],
        ocean=ocean, shape=shape, spacing_km=float(spacing_km),
        area_km2=float(spacing_km ** 2), land=land)

    if cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, x=grid.x, y=grid.y, latitude=grid.latitude,
            longitude=grid.longitude, ocean=grid.ocean,
            shape=np.array(shape), spacing_km=spacing_km,
            area_km2=grid.area_km2, land=grid.land)
    return grid


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

@dataclass
class ArcticGraph:
    """Neighbour structure over the ocean cells.

    Attributes
    ----------
    edge_index
        Shape ``(2, n_edges)``; ``edge_index[:, k]`` is the pair
        ``(source, target)`` of edge ``k``.  Edges are stored in both
        directions.
    edge_distance
        Length of each edge, km.
    edge_features
        Per-edge predictors used by the edge-conditioned network: normalised
        distance, the two components of the unit direction, and the difference
        in latitude.
    """

    edge_index: np.ndarray
    edge_distance: np.ndarray
    edge_features: np.ndarray
    n_nodes: int

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def degree(self) -> np.ndarray:
        """Number of incoming edges per node."""
        counts = np.bincount(self.edge_index[1], minlength=self.n_nodes)
        return counts.astype(float)


#: The eight index offsets of a cell's neighbours on the square grid.
NEIGHBOUR_OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                          if not (r == 0 and c == 0))


def classify_faces(grid: ArcticGrid) -> dict:
    """Sort each cell's faces into ocean, land and open boundary.

    Returns arrays of shape ``(n_cells, 8)``: the ocean index of the neighbour
    where there is one and minus one otherwise, together with boolean masks
    marking faces shared with land and faces on the open edge of the domain.
    The distinction is what keeps the transport honest.  A coast must reflect,
    so nothing may cross a land face, whereas the southern edge of the domain is
    open water and sea ice carried across it has genuinely left the Arctic.
    Treating the two alike either destroys ice along every coastline or dams the
    export that removes about a tenth of the ice volume each year.
    """
    rows, columns = grid.shape
    lookup = np.full(rows * columns, -1, np.int64)
    lookup[grid.ocean] = np.arange(grid.n_cells)

    row_index, column_index = np.divmod(grid.ocean, columns)
    neighbour = np.full((grid.n_cells, len(NEIGHBOUR_OFFSETS)), -1, np.int64)
    is_land = np.zeros_like(neighbour, bool)
    is_open = np.zeros_like(neighbour, bool)

    for k, (step_row, step_column) in enumerate(NEIGHBOUR_OFFSETS):
        near_row = row_index + step_row
        near_column = column_index + step_column
        in_bounds = ((near_row >= 0) & (near_row < rows)
                     & (near_column >= 0) & (near_column < columns))
        flat = np.where(in_bounds, near_row * columns + near_column, 0)

        found = np.where(in_bounds, lookup[flat], -1)
        land_here = in_bounds & (found < 0) & grid.land[flat]

        neighbour[:, k] = found
        is_land[:, k] = land_here
        is_open[:, k] = (found < 0) & ~land_here

    return {"neighbour": neighbour, "land": is_land, "open": is_open,
            "offsets": np.array(NEIGHBOUR_OFFSETS, float)}


def distance_to_coast(grid: ArcticGrid) -> np.ndarray:
    """Distance from each ocean cell to the nearest land, in cell widths.

    Computed by repeated dilation of the coastal cells over the graph of ocean
    neighbours, which is enough for the few cells over which the drift is
    tapered.
    """
    faces = classify_faces(grid)
    distance = np.where(faces["land"].any(axis=1), 1.0, np.inf)

    neighbour = faces["neighbour"]
    valid = neighbour >= 0
    for _ in range(8):
        candidate = np.where(valid, distance[np.where(valid, neighbour, 0)],
                             np.inf) + 1.0
        updated = np.minimum(distance, candidate.min(axis=1))
        if np.array_equal(updated, distance):
            break
        distance = updated
    return np.where(np.isfinite(distance), distance, 9.0)


def build_graph(grid: ArcticGrid, radius_km: float | None = None,
                include_self: bool = True) -> ArcticGraph:
    """Connect every ocean cell to its neighbours within ``radius_km``.

    A radius slightly above the diagonal spacing gives the eight nearest
    neighbours where they exist and fewer along coastlines, so the graph carries
    the shape of the basin.  The neighbourhood is deliberately local: the
    emulator is asked to discover the range over which influence travels by
    composing several rounds of message passing, and wiring long edges in by
    hand would answer that question in advance.

    Parameters
    ----------
    radius_km
        Connection radius.  Defaults to 1.5 times the grid spacing.
    include_self
        Add a self edge at every node so that a cell retains its own state.
    """
    if radius_km is None:
        radius_km = 1.5 * grid.spacing_km

    points = np.stack([grid.x, grid.y], axis=1)
    try:
        from scipy.spatial import cKDTree
        pairs = cKDTree(points).query_pairs(radius_km, output_type="ndarray")
        sources = np.concatenate([pairs[:, 0], pairs[:, 1]])
        targets = np.concatenate([pairs[:, 1], pairs[:, 0]])
    except ImportError:                             # pragma: no cover
        separation = np.hypot(grid.x[:, None] - grid.x[None, :],
                              grid.y[:, None] - grid.y[None, :])
        sources, targets = np.where((separation < radius_km) & (separation > 0))

    if include_self:
        loop = np.arange(grid.n_cells)
        sources = np.concatenate([sources, loop])
        targets = np.concatenate([targets, loop])

    edge_index = np.stack([sources, targets]).astype(np.int64)
    distance = grid.distance_km(edge_index[0], edge_index[1])

    delta_x = grid.x[edge_index[0]] - grid.x[edge_index[1]]
    delta_y = grid.y[edge_index[0]] - grid.y[edge_index[1]]
    length = np.maximum(distance, 1e-9)
    features = np.stack([
        distance / grid.spacing_km,
        delta_x / length,
        delta_y / length,
        grid.latitude[edge_index[0]] - grid.latitude[edge_index[1]],
    ], axis=1)

    return ArcticGraph(edge_index=edge_index, edge_distance=distance,
                       edge_features=features.astype(np.float32),
                       n_nodes=grid.n_cells)
