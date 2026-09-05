"""Spatially resolved pan-Arctic sea ice and ocean model.

Every ocean cell carries the same two variable column physics as
:mod:`arcticgnn.physics.seaice`, with three additions that couple the cells:

1. Atmospheric heat transport, as a diffusive convergence of the surface
   temperature field.
2. Sea ice motion, advected by a prescribed non-divergent drift representing the
   Beaufort Gyre and the Transpolar Drift.
3. Ocean transport of the sub-halocline layer, advected by a prescribed cyclonic
   boundary circulation and supplied with heat where Atlantic Water enters
   through the Nordic Seas.

The point of building this model is not realism for its own sake.  It creates a
system in which the distance over which one location influences another is set
by parameters that are known exactly, since they were prescribed.  The influence
length scale can therefore be measured directly, by perturbing one cell and
watching where the response appears, and the value recovered from the attention
weights of a trained network can be checked against it.  Without a system whose
answer is known in advance, an attention length scale is a number with nothing to
compare it to.

Transport is discretised on the same graph the emulators use.  Advection uses
first-order upwinding on the edges, which is diffusive but positivity preserving
and, importantly, never produces the spurious oscillations that a centred scheme
gives near the ice edge, where gradients are sharp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CONDUCTIVITY_ICE, LATENT_ICE
from ..physics.seaice import SeaIceParams
from .grid import (DOMAIN_LATITUDE, ArcticGraph, ArcticGrid,
                   classify_faces, distance_to_coast)
from ..physics.insolation import surface_shortwave

#: Centre of the Beaufort Gyre, km in the polar projection.
BEAUFORT_CENTRE = (-1150.0, -650.0)
#: Radius of the Beaufort Gyre, km.
BEAUFORT_RADIUS = 900.0
#: Longitude of the Fram Strait, degrees east, where Atlantic Water enters and
#: sea ice leaves the basin.
FRAM_LONGITUDE = 0.0


@dataclass(frozen=True)
class TransportParams:
    """Parameters of the horizontal coupling.

    The three length scales implied by these numbers are what the emulator is
    asked to recover.
    """

    #: Strength of the diffusive atmospheric heat convergence, W m^-2 K^-1.
    #: A surface temperature contrast of one kelvin between neighbouring cells
    #: drives this heat flux between them.
    atmosphere_coupling: float = 1.60

    #: Peak sea ice drift speed, km per year.  Typical Arctic drift is a few
    #: centimetres per second, which is of order one thousand kilometres a year.
    ice_drift_speed: float = 1100.0

    #: Peak speed of the sub-halocline circulation, km per year.
    ocean_speed: float = 550.0

    #: Diffusivity of the sub-halocline layer, expressed as the fraction of the
    #: temperature contrast between neighbours mixed away each year.
    ocean_diffusion: float = 0.55

    #: Rate at which the sub-halocline layer is restored towards the inflow
    #: temperature in the Atlantic Water entry region, per year.
    inflow_restoring: float = 1.10

    #: Half width of the Atlantic Water entry region, degrees of longitude.
    inflow_width: float = 45.0

    #: Southern edge of the Atlantic Water entry region, degrees north.
    inflow_latitude: float = 78.0

    #: Surface heat flux supplied by the warm Atlantic inflow in the Nordic and
    #: Barents Seas, W m^-2.  Without it the model freezes the whole northern
    #: North Atlantic in winter.  A single background ocean flux cannot express
    #: the strong zonal asymmetry of the real winter ice edge, which reaches far
    #: south in the Labrador Sea and the Sea of Okhotsk yet stops near Svalbard
    #: on the Atlantic side.  That asymmetry is caused by this inflow and is why
    #: those seas stay open through the winter.
    nordic_heat_flux: float = 46.0

    #: Western and eastern limits of the inflow sector, degrees of longitude.
    nordic_longitude: tuple = (-25.0, 60.0)

    #: Northern limit of the inflow, degrees north, and the width in degrees
    #: over which it is tapered away.  Beyond it the inflow has given up its
    #: heat and submerged beneath the halocline.
    nordic_latitude: float = 76.0
    nordic_taper: float = 3.0

    #: Thickness above which convergent drift is taken to ridge rather than to
    #: thicken the cell further, m.  The model carries one thickness per cell and
    #: no ice strength, so nothing in it resists convergence and drift piling
    #: against a coast would otherwise stack ice without limit.  Real convergence
    #: builds pressure ridges whose keels melt quickly from below and which are
    #: exported along the coast, so the excess is relaxed away on the timescale
    #: below.  The threshold is set near the thickest ridged ice observed north
    #: of Greenland and the Canadian Arctic Archipelago.
    ridging_thickness: float = 6.0

    #: Timescale on which thickness above the ridging threshold is removed, yr.
    ridging_timescale: float = 0.10

    #: Distance from the coast, in cell widths, over which the sea ice drift is
    #: brought to rest.  The prescribed drift is not tangential to the coastline,
    #: so without this the flow runs straight into the shore.  Slowing the ice
    #: as it nears land reproduces the landfast belt and, more to the point,
    #: removes the choice between destroying ice at every coast and piling it up
    #: without limit.
    coastal_taper_cells: float = 1.5


# --------------------------------------------------------------------------- #
# Prescribed circulation
# --------------------------------------------------------------------------- #

def ice_drift(grid: ArcticGrid, params: TransportParams
              ) -> tuple[np.ndarray, np.ndarray]:
    """Sea ice drift velocity at each cell, km per year.

    Built from a streamfunction so that the field is non-divergent and ice
    volume is conserved by the advection.  Two features are represented: the
    anticyclonic Beaufort Gyre over the Canada Basin, and the Transpolar Drift
    that carries ice from the Siberian shelves across the pole and out through
    the Fram Strait.
    """
    x, y = grid.x, grid.y

    # Beaufort Gyre: a Gaussian bump in the streamfunction gives closed
    # anticyclonic circulation around its centre.
    dx = x - BEAUFORT_CENTRE[0]
    dy = y - BEAUFORT_CENTRE[1]
    gyre = np.exp(-(dx ** 2 + dy ** 2) / BEAUFORT_RADIUS ** 2)
    gyre_u = -2.0 * dy / BEAUFORT_RADIUS ** 2 * gyre
    gyre_v = 2.0 * dx / BEAUFORT_RADIUS ** 2 * gyre

    # Transpolar Drift: a broad flow towards the Fram Strait, which lies at the
    # Greenwich meridian, in the direction of decreasing y in this projection.
    reach = np.radians(90.0 - DOMAIN_LATITUDE) * 6371.0
    transpolar = np.exp(-(x / (0.9 * reach)) ** 2)
    drift_u = np.zeros_like(x)
    drift_v = -transpolar / BEAUFORT_RADIUS

    u = gyre_u + drift_u
    v = gyre_v + drift_v
    speed = np.hypot(u, v)
    scale = params.ice_drift_speed / max(float(speed.max()), 1e-12)

    taper = np.tanh(distance_to_coast(grid) / params.coastal_taper_cells)
    return u * scale * taper, v * scale * taper


def ocean_flow(grid: ArcticGrid, params: TransportParams
               ) -> tuple[np.ndarray, np.ndarray]:
    """Sub-halocline circulation, km per year.

    A single cyclonic cell approximating the boundary current that carries
    Atlantic Water around the basin margins, strongest away from the pole.
    """
    x, y = grid.x, grid.y
    radius = np.hypot(x, y)
    reach = np.radians(90.0 - DOMAIN_LATITUDE) * 6371.0
    envelope = np.exp(-((radius - 0.72 * reach) / (0.42 * reach)) ** 2)

    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(radius > 1e-9, -y / np.maximum(radius, 1e-9), 0.0) * envelope
        v = np.where(radius > 1e-9, x / np.maximum(radius, 1e-9), 0.0) * envelope

    speed = np.hypot(u, v)
    scale = params.ocean_speed / max(float(speed.max()), 1e-12)
    return u * scale, v * scale


def nordic_mask(grid: ArcticGrid, params: TransportParams) -> np.ndarray:
    """Weight of the warm Atlantic surface inflow at each cell, from 0 to 1.

    Confined in longitude to the sector containing the Nordic and Barents Seas,
    with a smooth edge, and tapered towards the north.
    """
    longitude = (grid.longitude + 180.0) % 360.0 - 180.0
    west, east = params.nordic_longitude

    inside = (longitude >= west) & (longitude <= east)
    distance = np.where(longitude < west, west - longitude, longitude - east)
    sector = np.where(inside, 1.0, np.exp(-(np.maximum(distance, 0.0) / 8.0) ** 2))

    northern = 1.0 / (1.0 + np.exp((grid.latitude - params.nordic_latitude)
                                   / params.nordic_taper))
    return sector * northern


def inflow_mask(grid: ArcticGrid, params: TransportParams) -> np.ndarray:
    """Weight of the Atlantic Water entry region at each cell, from 0 to 1."""
    longitude = (grid.longitude - FRAM_LONGITUDE + 180.0) % 360.0 - 180.0
    near_gateway = np.exp(-(longitude / params.inflow_width) ** 2)
    outer = 1.0 / (1.0 + np.exp((grid.latitude - params.inflow_latitude) / 1.5))
    return near_gateway * outer


# --------------------------------------------------------------------------- #
# Transport operators
# --------------------------------------------------------------------------- #

def _boundary_outflow(grid: ArcticGrid, u: np.ndarray, v: np.ndarray
                      ) -> np.ndarray:
    """Rate at which each cell loses material across the open boundary, per year.

    Only faces on the open edge of the domain count.  Faces shared with land are
    closed, and the drift has already been brought to rest near them, so nothing
    is lost to a coastline.
    """
    faces = classify_faces(grid)
    offsets = faces["offsets"]
    outflow = np.zeros(grid.n_cells)

    for k in range(offsets.shape[0]):
        open_face = faces["open"][:, k]
        if not open_face.any():
            continue
        step = offsets[k]
        norm = float(np.hypot(step[0], step[1]))
        normal_x, normal_y = step[0] / norm, step[1] / norm
        speed_out = (u * normal_x + v * normal_y) / (grid.spacing_km * norm)
        outflow += np.where(open_face, np.maximum(speed_out, 0.0), 0.0)

    return outflow


class GraphTransport:
    """Upwind advection and diffusion on the cell graph.

    The operators are assembled once as sparse matrices and reused at every
    step, since the grid and the prescribed velocities never change.
    """

    def __init__(self, grid: ArcticGrid, graph: ArcticGraph,
                 u: np.ndarray, v: np.ndarray):
        from scipy import sparse

        source, target = graph.edge_index
        interior = source != target
        source, target = source[interior], target[interior]

        # Outward unit vector from the target cell towards the source cell.
        dx = grid.x[source] - grid.x[target]
        dy = grid.y[source] - grid.y[target]
        length = np.hypot(dx, dy)
        nx, ny = dx / length, dy / length

        # Velocity at the face between the two cells, projected onto the face
        # normal.  A positive value carries material from the source into the
        # target.
        face_u = 0.5 * (u[source] + u[target])
        face_v = 0.5 * (v[source] + v[target])
        inflow = (face_u * nx + face_v * ny) / length

        self.n = grid.n_cells
        # Upwinding: a cell gains from a neighbour only where the face velocity
        # points towards it, and loses to that neighbour otherwise.
        gain = np.maximum(inflow, 0.0)
        loss = np.maximum(-inflow, 0.0)

        self.advection = (
            sparse.coo_matrix((gain, (target, source)), shape=(self.n, self.n))
            - sparse.coo_matrix((loss, (target, target)), shape=(self.n, self.n))
        ).tocsr()

        # Faces that a cell shares with land or with the world outside the
        # domain carry material out but never bring any in.  Without this term
        # everything advected towards a coast simply accumulates there, and the
        # drift piles sea ice against northern Greenland to tens of metres while
        # the export through the Fram Strait, which removes roughly a tenth of
        # the Arctic ice volume every year, never happens at all.
        self.boundary_loss = _boundary_outflow(grid, u, v)
        self.advection = (self.advection
                          - sparse.diags(self.boundary_loss)).tocsr()

        weight = 1.0 / length ** 2
        neighbour_sum = np.bincount(target, weights=weight, minlength=self.n)
        normalised = weight / np.maximum(neighbour_sum[target], 1e-12)
        # After normalisation each row averages its neighbours and subtracts the
        # cell itself, so the coefficient is the fraction of a neighbour
        # contrast mixed per unit time whatever the number of neighbours.
        self.diffusion = (
            sparse.coo_matrix((normalised, (target, source)),
                              shape=(self.n, self.n))
            - sparse.eye(self.n)
        ).tocsr()

    def advect(self, field: np.ndarray) -> np.ndarray:
        """Convergence of the advective flux of ``field``."""
        return self.advection.dot(field)

    def diffuse(self, field: np.ndarray) -> np.ndarray:
        """Diffusive convergence of ``field``."""
        return self.diffusion.dot(field)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

class SpatialArcticModel:
    """Pan-Arctic sea ice and ocean model on the polar grid."""

    def __init__(self, grid: ArcticGrid, graph: ArcticGraph,
                 column: SeaIceParams | None = None,
                 transport: TransportParams | None = None):
        self.grid = grid
        self.graph = graph
        self.p = column if column is not None else SeaIceParams()
        self.t = transport if transport is not None else TransportParams()

        self.ice_u, self.ice_v = ice_drift(grid, self.t)
        self.ocean_u, self.ocean_v = ocean_flow(grid, self.t)
        self.ice_transport = GraphTransport(grid, graph, self.ice_u, self.ice_v)
        self.ocean_transport = GraphTransport(grid, graph, self.ocean_u,
                                              self.ocean_v)
        self.inflow = inflow_mask(grid, self.t)
        self.nordic = nordic_mask(grid, self.t)

        self._season: dict = {}
        self._cache_constants()

    def _cache_constants(self) -> None:
        p = self.p
        self._inv_albedo_e = 1.0 / (LATENT_ICE * p.albedo_scale)
        self._inv_switch_e = 1.0 / (LATENT_ICE * p.switch_scale)
        self._albedo_range = p.albedo_ocean - p.albedo_ice
        self._inv_latent = 1.0 / LATENT_ICE
        self._inv_c_mixed = 1.0 / p.c_mixed
        self._inv_c_deep = 1.0 / p.c_deep
        self._melt_cap_sq = p.melt_cap_scale ** 2
        self._feedback_range = p.turbulent_feedback - p.longwave_feedback
        self._thickness_round = LATENT_ICE * 0.02
        self._t_atlantic = (p.atlantic_temperature
                            + p.atlantic_sensitivity * p.forcing)

    # -- forcing -------------------------------------------------------------- #

    def _season_table(self, steps_per_year: int) -> np.ndarray:
        """Surface shortwave on the half-step grid, shape ``(2 n, cells)``."""
        table = self._season.get(steps_per_year)
        if table is None:
            times = np.arange(2 * steps_per_year) / (2.0 * steps_per_year)
            table = surface_shortwave(self.grid.latitude, times)
            self._season[steps_per_year] = table
        return table

    # -- tendency ------------------------------------------------------------- #

    def tendency(self, shortwave: np.ndarray, phase: float,
                 state: np.ndarray) -> np.ndarray:
        """Time derivative of the state, shape ``(cells, 2)``, per year."""
        p = self.p
        enthalpy = state[:, 0]
        deep = state[:, 1]

        open_fraction = 0.5 * (1.0 + np.tanh(enthalpy * self._inv_albedo_e))
        albedo = p.albedo_ice + self._albedo_range * open_fraction
        upward = p.longwave_offset + p.longwave_annual * phase
        net_atmospheric = (1.0 - albedo) * shortwave - upward + p.forcing

        feedback = p.longwave_feedback + self._feedback_range * open_fraction

        thickness = (0.5 * (np.sqrt(enthalpy * enthalpy
                                    + self._thickness_round ** 2) - enthalpy)
                     * self._inv_latent + p.thickness_floor)
        ice_branch = net_atmospheric / (p.longwave_feedback
                                        + CONDUCTIVITY_ICE / thickness)
        ice_branch = 0.5 * (ice_branch - np.sqrt(ice_branch * ice_branch
                                                 + self._melt_cap_sq))
        switch = 0.5 * (1.0 + np.tanh(enthalpy * self._inv_switch_e))
        temperature = (switch * enthalpy * self._inv_c_mixed
                       + (1.0 - switch) * ice_branch)

        mixing = p.mixing_ice + p.mixing_open * open_fraction

        # Column budget, with the warm Atlantic surface inflow added where it
        # reaches.
        surface = (net_atmospheric - feedback * temperature
                   + p.ocean_heat_flux + mixing * deep
                   + self.t.nordic_heat_flux * self.nordic)

        # Atmospheric heat transport acts on the surface temperature contrast
        # between neighbours.
        surface = surface + self.t.atmosphere_coupling * \
            self.ice_transport.diffuse(temperature)

        # Sea ice motion.  Only the frozen part of the enthalpy is carried by
        # the ice; the open-ocean part belongs to the mixed layer and stays put.
        frozen = np.minimum(enthalpy, 0.0)
        surface = surface + self.ice_transport.advect(frozen)

        # Ridging closure: thickness beyond the threshold is relaxed away.
        excess = np.maximum(thickness - self.t.ridging_thickness, 0.0)
        surface = surface + excess * LATENT_ICE / self.t.ridging_timescale

        interior = ((p.ventilation * (self._t_atlantic - deep) - mixing * deep)
                    * self._inv_c_deep)
        interior = interior + self.ocean_transport.advect(deep)
        interior = interior + self.t.ocean_diffusion * \
            self.ocean_transport.diffuse(deep)
        interior = interior + (self.t.inflow_restoring * self.inflow
                               * (self._t_atlantic - deep))

        return np.stack([surface, interior], axis=-1)

    # -- integration ---------------------------------------------------------- #

    def integrate(self, state: np.ndarray, years: float,
                  steps_per_year: int = 200, t0: float = 0.0,
                  store_every: int = 0, noise: float = 0.0,
                  seed: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the field forward, returning ``(times, states)``."""
        state = np.asarray(state, float).copy()
        dt = 1.0 / steps_per_year
        n_steps = int(round(years * steps_per_year))
        table = self._season_table(steps_per_year)
        n_half = 2 * steps_per_year
        offset = int(round((t0 % 1.0) * n_half))
        rng = np.random.default_rng(seed) if noise > 0 else None
        noise_scale = noise * np.sqrt(dt)

        phases = -np.cos(np.pi * np.arange(n_half) / steps_per_year)

        keep = store_every > 0
        times: list[float] = []
        states: list[np.ndarray] = []
        if keep:
            times.append(t0)
            states.append(state.copy())

        for i in range(n_steps):
            i0 = (offset + 2 * i) % n_half
            i1 = (offset + 2 * i + 1) % n_half
            i2 = (offset + 2 * i + 2) % n_half

            k1 = self.tendency(table[i0], phases[i0], state)
            k2 = self.tendency(table[i1], phases[i1], state + 0.5 * dt * k1)
            k3 = self.tendency(table[i1], phases[i1], state + 0.5 * dt * k2)
            k4 = self.tendency(table[i2], phases[i2], state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if rng is not None:
                state[:, 0] += rng.normal(0.0, noise_scale, size=state.shape[0])
            if keep and (i + 1) % store_every == 0:
                times.append(t0 + (i + 1) * dt)
                states.append(state.copy())

        if not keep:
            return np.array([t0 + n_steps * dt]), state[None, ...]
        return np.array(times), np.array(states)

    # -- diagnostics ---------------------------------------------------------- #

    def thickness(self, enthalpy: np.ndarray) -> np.ndarray:
        """Ice thickness, m."""
        return np.maximum(0.0, -np.asarray(enthalpy, float)) / LATENT_ICE

    def concentration(self, enthalpy: np.ndarray) -> np.ndarray:
        """Ice covered fraction of a cell, from the same smooth switch that
        governs the albedo, so that cover and albedo stay consistent."""
        return 1.0 - 0.5 * (1.0 + np.tanh(np.asarray(enthalpy, float)
                                          * self._inv_albedo_e))

    def extent_km2(self, enthalpy: np.ndarray, threshold: float = 0.15
                   ) -> np.ndarray:
        """Sea ice extent, km^2, using the conventional concentration cutoff.

        Extent counts the whole area of every cell whose ice concentration
        exceeds the threshold, which is how the satellite record defines it.
        """
        cover = self.concentration(enthalpy)
        return (cover >= threshold).sum(axis=-1) * self.grid.area_km2

    def area_km2(self, enthalpy: np.ndarray) -> np.ndarray:
        """Sea ice area, km^2, weighting each cell by its concentration."""
        return self.concentration(enthalpy).sum(axis=-1) * self.grid.area_km2

    def volume_km3(self, enthalpy: np.ndarray) -> np.ndarray:
        """Sea ice volume, km^3."""
        return (self.thickness(enthalpy) * 1e-3).sum(axis=-1) * self.grid.area_km2

    def initial_state(self, thickness_m: float = 2.0,
                      deep_temperature: float = 0.6) -> np.ndarray:
        """A uniform starting field."""
        n = self.grid.n_cells
        return np.stack([np.full(n, -thickness_m * LATENT_ICE),
                         np.full(n, deep_temperature)], axis=-1)

    def with_forcing(self, forcing: float) -> "SpatialArcticModel":
        """A copy of the model at a different greenhouse forcing."""
        clone = SpatialArcticModel.__new__(SpatialArcticModel)
        clone.__dict__.update(self.__dict__)
        clone.p = self.p.replace(forcing=float(forcing))
        clone._season = self._season
        clone._cache_constants()
        return clone
