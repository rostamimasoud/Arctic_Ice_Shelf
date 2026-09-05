"""Seasonal Arctic sea-ice and ocean column model.

The model carries two prognostic variables and is forced by a prescribed
seasonal cycle of insolation:

``E``
    Surface enthalpy in W yr m^-2.  Negative values describe an ice cover of
    thickness ``h = -E / LATENT_ICE``; positive values describe a mixed layer
    warmer than freezing by ``E / c_ml``.
``T_d``
    Temperature of the sub-halocline water, in K above the freezing point.

Two positive feedbacks act on the surface budget.  The first is the classical
ice-albedo feedback: thinning ice exposes darker ocean, which absorbs more
shortwave radiation and thins the ice further.  The second is a subsurface heat
feedback that is the reason this model has two variables instead of one.  Under
a thick ice cover the halocline suppresses vertical mixing, so heat carried into
the Arctic by Atlantic Water accumulates below the mixed layer.  When the ice
thins, wind stress acts on open water, mixing strengthens, and the stored heat
is released upward, thinning the ice further still.  The strength of the second
feedback therefore depends on how much heat has accumulated, which is a slow
variable, and this is what allows the system to hold two distinct annual cycles
at the same greenhouse forcing.

Both feedbacks are written with smooth switch functions.  This is a requirement
rather than a convenience: the continuation and fold-refinement machinery in
:mod:`arcticgnn.dynsys.continuation` differentiates the one-year map by finite
differences, and a hard ``min`` or a piecewise branch on the sign of ``E``
introduces a kink that makes the Jacobian meaningless in exactly the region near
the fold where it matters most.

The right-hand side is vectorised over a leading ensemble axis.  A finite
difference Jacobian of the annual map then costs a single batched integration
instead of one integration per column, which is what makes continuation of the
stroboscopic map affordable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..config import (CONDUCTIVITY_ICE, LATENT_ICE, heat_capacity)
from .insolation import surface_shortwave

#: Number of prognostic variables.
NDIM = 2

#: Names of the control parameters accepted by :meth:`SeaIceParams.with_control`.
CONTROL_PARAMETERS = ("forcing", "atlantic_temperature", "ventilation",
                      "ocean_heat_flux", "mixing_open")


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SeaIceParams:
    """Parameters of the column model.

    Default values describe the central Arctic Ocean.  The two quantities set by
    :mod:`arcticgnn.physics.calibrate` so that the control integration
    reproduces the observed seasonal cycle of ice thickness.
    """

    # -- shortwave ---------------------------------------------------------- #
    #: Latitude the column represents, degrees north.  Shortwave forcing is
    #: computed from orbital geometry at this latitude instead of being
    #: prescribed as a fitted cycle.  The distinction matters more than it
    #: looks: the real polar summer is far brighter and far shorter than any
    #: low-order Fourier fit suggests, and because the ice surface sits at the
    #: melting point through that period, every extra watt goes into melting
    #: instead of warming.  A fitted cycle with the correct annual mean but a
    #: gentler peak therefore understates summer melt badly, and the energy
    #: budget calibrated against it comes out unphysical.
    latitude: float = 85.0

    # -- surface albedo ----------------------------------------------------- #
    albedo_ice: float = 0.68
    albedo_ocean: float = 0.20
    #: Ice thickness over which the surface darkens, m.  This aggregates melt
    #: pond growth and the opening of the pack, both of which set in well before
    #: the ice disappears.
    albedo_scale: float = 1.00

    # -- outgoing longwave and turbulent fluxes ----------------------------- #
    #: Temperature-independent part of the net upward surface flux, W m^-2.
    longwave_offset: float = 66.0
    #: Sensitivity of the net upward surface flux to surface temperature over a
    #: closed ice cover, W m^-2 K^-1.  This is essentially the local Planck
    #: response, because turbulent exchange across an ice surface is weak.
    longwave_feedback: float = 2.90
    #: The same sensitivity over open water, W m^-2 K^-1.  It is several times
    #: larger, because sensible and latent heat exchange with the atmosphere
    #: scale steeply with the air to sea temperature difference and are very
    #: efficient once the insulating ice lid is removed.  Keeping the two
    #: sensitivities distinct is what allows the model to refreeze in winter
    #: after a summer without ice, in agreement with the rapid recovery seen
    #: when sea ice is artificially removed from comprehensive climate models.
    turbulent_feedback: float = 12.0
    #: Seasonal amplitude of the net upward surface flux, W m^-2.  The sign is
    #: negative because the net loss is largest in winter: although the surface
    #: is then coldest, the overlying atmosphere is colder and much drier still,
    #: so the downward longwave flux falls faster than the upward one.  A
    #: linearisation in surface temperature alone cannot reproduce that, which is
    #: why this seasonal term carries it explicitly.
    #:
    #: This value is not fitted.  It follows from the observed non-solar surface
    #: fluxes over Arctic sea ice, about 27 W m^-2 of net loss in midwinter with
    #: a surface near 30 K below freezing and about 15 W m^-2 in midsummer with
    #: the surface at the melting point.  Together with the flux sensitivity
    #: those two figures fix both the offset and this amplitude.  Fitting the
    #: amplitude instead is a trap: the thickness targets can also be met by
    #: driving it towards zero, which balances the budget only by having the
    #: winter surface gain heat, and no amount of agreement with the observed
    #: thickness makes that an Arctic.
    longwave_annual: float = -49.5

    # -- radiative forcing -------------------------------------------------- #
    #: Greenhouse forcing anomaly applied to the surface budget, W m^-2.
    forcing: float = 0.0

    # -- ocean --------------------------------------------------------------- #
    mixed_layer_depth: float = 75.0
    deep_layer_depth: float = 500.0
    #: Background heat flux from the ocean into the ice base, W m^-2.
    ocean_heat_flux: float = 2.0
    #: Mixing coefficient across the halocline under a closed ice cover,
    #: W m^-2 K^-1.
    mixing_ice: float = 0.55
    #: Additional mixing over fully open water, W m^-2 K^-1.
    mixing_open: float = 5.20
    #: Ventilation rate of the sub-halocline layer by Atlantic Water inflow,
    #: W m^-2 K^-1.
    ventilation: float = 2.20
    #: Temperature of the inflowing Atlantic Water, K above freezing.
    atlantic_temperature: float = 2.40
    #: Warming of the Atlantic inflow per unit greenhouse forcing, K (W m^-2)^-1.
    atlantic_sensitivity: float = 0.42

    # -- numerics ------------------------------------------------------------ #
    #: Width of the switch between the ice-covered and ice-free surface
    #: temperature branches, expressed as an equivalent ice thickness in m.
    switch_scale: float = 0.04
    #: Floor on ice thickness in the conduction term, m.
    thickness_floor: float = 0.01
    #: Rounding scale of the surface melting cap, K.
    melt_cap_scale: float = 0.05

    name: str = "central_arctic"

    # -- derived ------------------------------------------------------------- #

    @property
    def c_mixed(self) -> float:
        """Heat capacity of the mixed layer, W yr m^-2 K^-1."""
        return heat_capacity(self.mixed_layer_depth)

    @property
    def c_deep(self) -> float:
        """Heat capacity of the sub-halocline layer, W yr m^-2 K^-1."""
        return heat_capacity(self.deep_layer_depth)

    def with_control(self, **kwargs) -> "SeaIceParams":
        """Return a copy with one or more control parameters replaced."""
        unknown = set(kwargs) - set(CONTROL_PARAMETERS)
        if unknown:
            raise ValueError(
                f"unknown control parameter(s) {sorted(unknown)}; "
                f"expected a subset of {CONTROL_PARAMETERS}")
        # Control parameters may be arrays: a whole sweep is integrated as one
        # batch, and the continuation routines vary the parameter across the
        # batch to build a Jacobian.  Scalars are still normalised to float so
        # that the seasonal table stays cacheable.
        converted = {k: (float(v) if np.isscalar(v) or np.ndim(v) == 0
                         else np.asarray(v, float))
                     for k, v in kwargs.items()}
        return replace(self, **converted)

    def replace(self, **kwargs) -> "SeaIceParams":
        """Return a copy with any field replaced."""
        return replace(self, **kwargs)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

@dataclass
class ColumnDiagnostics:
    """Instantaneous diagnostics of a column state."""

    thickness: np.ndarray          # m
    open_fraction: np.ndarray      # 0 closed ice cover, 1 ice free
    surface_temperature: np.ndarray  # K above freezing
    albedo: np.ndarray
    absorbed_shortwave: np.ndarray  # W m^-2
    basal_flux: np.ndarray         # W m^-2, ocean into ice base
    mixing: np.ndarray             # W m^-2 K^-1
    deep_temperature: np.ndarray   # K above freezing


# --------------------------------------------------------------------------- #
# Smooth switches
# --------------------------------------------------------------------------- #

def _smooth_min_zero(x: np.ndarray, scale: float) -> np.ndarray:
    """Smooth approximation to ``min(0, x)`` with rounding width ``scale``."""
    return 0.5 * (x - np.sqrt(x * x + scale * scale))


def _ramp(x: np.ndarray, scale: float) -> np.ndarray:
    """Smooth step from 0 to 1 as ``x`` rises through zero over width ``scale``."""
    return 0.5 * (1.0 + np.tanh(x / scale))


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

class SeaIceColumn:
    """Seasonal sea-ice and ocean column model.

    All methods accept either a single state of shape ``(2,)`` or an ensemble of
    shape ``(n, 2)`` and return correspondingly shaped output.
    """

    def __init__(self, params: SeaIceParams | None = None):
        self.p = params if params is not None else SeaIceParams()
        self._cache_constants()

    def _cache_constants(self) -> None:
        """Precompute the combinations of parameters used on the hot path."""
        p = self.p
        self._inv_albedo_e = 1.0 / (LATENT_ICE * p.albedo_scale)
        self._inv_switch_e = 1.0 / (LATENT_ICE * p.switch_scale)
        self._albedo_range = p.albedo_ocean - p.albedo_ice
        self._inv_latent = 1.0 / LATENT_ICE
        self._inv_c_mixed = 1.0 / p.c_mixed
        self._inv_c_deep = 1.0 / p.c_deep
        self._melt_cap_sq = p.melt_cap_scale ** 2
        self._feedback_range = p.turbulent_feedback - p.longwave_feedback
        # Rounding scale of the thickness entering the conduction term, in the
        # same units as the enthalpy.
        self._thickness_round = LATENT_ICE * 0.02
        self._t_atlantic = p.atlantic_temperature + p.atlantic_sensitivity * p.forcing
        self._season_key = (float(p.latitude),)

    #: Seasonal tables, shared across instances.  A ramped integration rebuilds
    #: the model object at every step as the control parameter moves, and none of
    #: the control parameters enter the seasonal cycle, so caching per instance
    #: would rebuild an identical table thousands of times.
    _SEASON_CACHE: dict = {}

    def _season_table(self, steps_per_year: int) -> tuple[np.ndarray, np.ndarray]:
        """Seasonal fluxes tabulated on the half-step grid of the integrator.

        Every Runge-Kutta stage falls on a multiple of half a step, and the
        seasonal cycle repeats each year, so the trigonometric terms can be
        evaluated once per grid resolution instead of four times per step.  This
        removes the dominant cost of long integrations.
        """
        key = self._season_key + (steps_per_year,)
        cached = self._SEASON_CACHE.get(key)
        if cached is not None:
            return cached
        p = self.p
        tau = np.arange(2 * steps_per_year) / (2.0 * steps_per_year)
        two_pi_t = 2.0 * np.pi * tau
        shortwave = surface_shortwave(np.array([p.latitude]), tau)[:, 0]
        # Only the shape of the seasonal cycle is tabulated.  Its amplitude is
        # applied in the tendency, so that the amplitude may be array valued and
        # differ between the configurations sharing one batched integration.
        longwave = -np.cos(two_pi_t)
        self._SEASON_CACHE[key] = (shortwave, longwave)
        return shortwave, longwave

    def _tendency(self, shortwave: float, longwave: float,
                  x: np.ndarray) -> np.ndarray:
        """Tendency given the seasonal fluxes already evaluated.

        This is the fused form of :meth:`rhs`.  The two smooth switches are
        evaluated once each and shared between the albedo, the surface
        temperature blend and the mixing coefficient, which the readable form
        below recomputes several times over.
        """
        p = self.p
        enthalpy = x[..., 0]
        deep = x[..., 1]

        open_fraction = 0.5 * (1.0 + np.tanh(enthalpy * self._inv_albedo_e))
        albedo = p.albedo_ice + self._albedo_range * open_fraction
        net_atmospheric = ((1.0 - albedo) * shortwave
                           - (p.longwave_offset + p.longwave_annual * longwave)
                           + p.forcing)

        feedback = p.longwave_feedback + self._feedback_range * open_fraction

        # Smooth counterpart of max(0, -E).  Using the exact expression puts a
        # kink at the freezing point, and because conduction goes as one over
        # thickness that kink is violent: the continuation differentiates a one
        # year integration, so its Jacobians would be corrupted in exactly the
        # region near the threshold where they matter.
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
        surface = (net_atmospheric - feedback * temperature
                   + p.ocean_heat_flux + mixing * deep)
        interior = ((p.ventilation * (self._t_atlantic - deep) - mixing * deep)
                    * self._inv_c_deep)

        return np.stack([surface, interior], axis=-1)

    # -- forcing ------------------------------------------------------------- #

    def shortwave(self, t: float | np.ndarray) -> np.ndarray:
        """Downward shortwave flux at the surface, W m^-2."""
        times = np.atleast_1d(np.asarray(t, float))
        values = surface_shortwave(np.array([self.p.latitude]), times)[:, 0]
        return values if np.ndim(t) else float(values[0])

    def net_upward_offset(self, t: float | np.ndarray) -> np.ndarray:
        """Temperature-independent part of the net upward surface flux, W m^-2."""
        p = self.p
        two_pi_t = 2.0 * np.pi * np.asarray(t, float)
        return p.longwave_offset - p.longwave_annual * np.cos(two_pi_t)

    def atlantic_temperature(self) -> float:
        """Atlantic inflow temperature including its response to forcing, K."""
        p = self.p
        return p.atlantic_temperature + p.atlantic_sensitivity * p.forcing

    # -- state maps ---------------------------------------------------------- #

    def thickness(self, enthalpy: np.ndarray) -> np.ndarray:
        """Ice thickness, m.  Zero where the surface enthalpy is positive."""
        e = np.asarray(enthalpy, float)
        return np.maximum(0.0, -e) / LATENT_ICE

    def open_fraction(self, enthalpy: np.ndarray) -> np.ndarray:
        """Smooth measure of exposure of the ocean surface, from 0 to 1."""
        e = np.asarray(enthalpy, float)
        return _ramp(e, LATENT_ICE * self.p.albedo_scale)

    def albedo(self, enthalpy: np.ndarray) -> np.ndarray:
        """Surface albedo, darkening smoothly as the ice thins."""
        p = self.p
        phi = self.open_fraction(enthalpy)
        return p.albedo_ice + (p.albedo_ocean - p.albedo_ice) * phi

    def mixing(self, enthalpy: np.ndarray) -> np.ndarray:
        """Halocline mixing coefficient, W m^-2 K^-1.

        This is the heart of the subsurface heat feedback.  Mixing strengthens
        as the ice cover opens, because wind stress then acts directly on the
        ocean surface and erodes the stratification that had been trapping heat
        at depth.
        """
        p = self.p
        return p.mixing_ice + p.mixing_open * self.open_fraction(enthalpy)

    def surface_temperature(self, enthalpy: np.ndarray,
                            t: float | np.ndarray) -> np.ndarray:
        """Surface temperature, K above freezing.

        Over open water the surface temperature is set by the mixed-layer
        enthalpy.  Over ice it is set by the balance between the net atmospheric
        flux and conduction through the ice, and is capped at the melting point,
        with any excess energy going into melting instead of warming.  The two
        branches both approach the freezing point as the ice vanishes, so they
        are blended with a narrow smooth switch.
        """
        p = self.p
        e = np.asarray(enthalpy, float)

        ocean_branch = e / p.c_mixed

        h = self.thickness(e) + p.thickness_floor
        net_atmospheric = ((1.0 - self.albedo(e)) * self.shortwave(t)
                           - self.net_upward_offset(t) + p.forcing)
        conduction = CONDUCTIVITY_ICE / h
        ice_branch = _smooth_min_zero(
            net_atmospheric / (p.longwave_feedback + conduction),
            p.melt_cap_scale)

        psi = _ramp(e, LATENT_ICE * p.switch_scale)
        return psi * ocean_branch + (1.0 - psi) * ice_branch

    def basal_flux(self, enthalpy: np.ndarray,
                   deep_temperature: np.ndarray) -> np.ndarray:
        """Heat flux from the ocean interior into the surface layer, W m^-2."""
        return (self.p.ocean_heat_flux
                + self.mixing(enthalpy) * np.asarray(deep_temperature, float))

    # -- tendencies ---------------------------------------------------------- #

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """Time derivative of the state, in units per year.

        Accepts shape ``(2,)`` or ``(n, 2)``.
        """
        x = np.asarray(x, float)
        return self._tendency(float(self.shortwave(t)),
                              float(-np.cos(2.0 * np.pi * t)), x)

    def diagnostics(self, x: np.ndarray,
                    t: float = 0.0) -> ColumnDiagnostics:
        """Diagnostics of a state at time ``t``."""
        x = np.atleast_2d(np.asarray(x, float))
        e, td = x[..., 0], x[..., 1]
        return ColumnDiagnostics(
            thickness=self.thickness(e),
            open_fraction=self.open_fraction(e),
            surface_temperature=self.surface_temperature(e, t),
            albedo=self.albedo(e),
            absorbed_shortwave=(1.0 - self.albedo(e)) * self.shortwave(t),
            basal_flux=self.basal_flux(e, td),
            mixing=self.mixing(e),
            deep_temperature=td,
        )

    # -- integration --------------------------------------------------------- #

    def step_rk4(self, t: float, x: np.ndarray, dt: float) -> np.ndarray:
        """One classical fourth-order Runge-Kutta step."""
        k1 = self.rhs(t, x)
        k2 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = self.rhs(t + dt, x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _step_tabulated(self, x: np.ndarray, dt: float, index: int,
                        shortwave: np.ndarray, longwave: np.ndarray,
                        n_half: int) -> np.ndarray:
        """One Runge-Kutta step using the tabulated seasonal cycle."""
        i0 = index % n_half
        i1 = (index + 1) % n_half
        i2 = (index + 2) % n_half
        s0, l0 = shortwave[i0], longwave[i0]
        s1, l1 = shortwave[i1], longwave[i1]
        s2, l2 = shortwave[i2], longwave[i2]

        k1 = self._tendency(s0, l0, x)
        k2 = self._tendency(s1, l1, x + 0.5 * dt * k1)
        k3 = self._tendency(s1, l1, x + 0.5 * dt * k2)
        k4 = self._tendency(s2, l2, x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate(self, x0: np.ndarray, years: float, steps_per_year: int = 400,
                  t0: float = 0.0, store_every: int = 0,
                  forcing: "callable | None" = None,
                  noise: float = 0.0, seed: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate with a fixed step, returning ``(t, x)``.

        A fixed step is used deliberately.  The continuation routines difference
        this map to build Jacobians, and the step-size control of an adaptive
        solver makes the map a slightly discontinuous function of its initial
        condition, which corrupts those differences.

        Parameters
        ----------
        x0
            Initial state, shape ``(2,)`` or ``(n, 2)``.
        years
            Length of the integration.
        steps_per_year
            Fixed steps per year.
        store_every
            Store every this many steps.  ``0`` stores only the final state.
        forcing
            Optional callable mapping time in years to a dictionary of control
            parameter values, applied before each step.
        noise
            Standard deviation of additive white noise on the surface enthalpy
            tendency, in W m^-2.  Applied as an Euler-Maruyama increment on top
            of the deterministic step.
        seed
            Seed for the noise.
        """
        x = np.asarray(x0, float).copy()
        dt = 1.0 / steps_per_year
        n_steps = int(round(years * steps_per_year))
        rng = np.random.default_rng(seed) if noise > 0 else None

        keep = store_every > 0
        times: list[float] = []
        states: list[np.ndarray] = []
        if keep:
            times.append(t0)
            states.append(x.copy())

        base = self.p
        model = self
        shortwave, longwave = self._season_table(steps_per_year)
        n_half = 2 * steps_per_year
        # The tabulated grid is indexed from the start of a calendar year, so an
        # integration starting mid-year must be offset accordingly.
        offset = int(round((t0 % 1.0) * n_half))
        noise_scale = noise * np.sqrt(dt)

        for i in range(n_steps):
            t = t0 + i * dt
            if forcing is not None:
                model = SeaIceColumn(base.with_control(**forcing(t)))
                shortwave, longwave = model._season_table(steps_per_year)
            x = model._step_tabulated(x, dt, offset + 2 * i,
                                      shortwave, longwave, n_half)
            if rng is not None:
                x[..., 0] += rng.normal(0.0, noise_scale, size=np.shape(x)[:-1])
            if keep and (i + 1) % store_every == 0:
                times.append(t + dt)
                states.append(x.copy())

        if not keep:
            return np.array([t0 + n_steps * dt]), x[None, ...]
        return np.array(times), np.array(states)

    # -- the annual map ------------------------------------------------------ #

    def annual_map(self, x: np.ndarray, steps_per_year: int = 400
                   ) -> np.ndarray:
        """Advance the state by exactly one year.

        A fixed point of this map is a periodic annual cycle of the forced
        system, which is the object whose branches and folds are continued.
        """
        return self.integrate(x, years=1.0, steps_per_year=steps_per_year)[1][-1]

    def annual_cycle(self, x0: np.ndarray, steps_per_year: int = 400,
                     store_every: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """One year of the trajectory starting from ``x0``."""
        return self.integrate(x0, years=1.0, steps_per_year=steps_per_year,
                              store_every=store_every)

    def spin_up(self, x0: np.ndarray, years: float = 500.0,
                steps_per_year: int = 400) -> np.ndarray:
        """Integrate long enough to settle onto an attractor."""
        return self.integrate(x0, years=years,
                              steps_per_year=steps_per_year)[1][-1]

    # -- attractors ---------------------------------------------------------- #

    def initial_state(self, regime: str) -> np.ndarray:
        """A state well inside the basin of the named regime.

        ``perennial``
            Thick ice surviving the summer, with little heat stored at depth.
        ``seasonal``
            Ice free in summer, with a warm sub-halocline reservoir.
        """
        if regime == "perennial":
            return np.array([-3.0 * LATENT_ICE, 0.15])
        if regime == "seasonal":
            return np.array([0.6 * self.p.c_mixed, 2.6])
        raise ValueError(
            f"regime must be 'perennial' or 'seasonal', got {regime!r}")

    def attractor(self, regime: str, years: float = 500.0,
                  steps_per_year: int = 400) -> np.ndarray:
        """Spin up from the named regime and return the state on 1 January."""
        return self.spin_up(self.initial_state(regime), years=years,
                            steps_per_year=steps_per_year)


# --------------------------------------------------------------------------- #
# Annual-cycle summaries
# --------------------------------------------------------------------------- #

@dataclass
class AnnualSummary:
    """Summary of one annual cycle of a column."""

    thickness_mean: float          # m
    thickness_max: float           # m, seasonal maximum, reached in spring
    thickness_min: float           # m, seasonal minimum, reached in autumn
    ice_free_fraction: float       # fraction of the year with no ice
    deep_temperature_mean: float   # K above freezing
    open_fraction_mean: float
    perennial: bool                # ice survives the whole year


#: Ice thinner than this is treated as absent when summarising a cycle, m.
ICE_FREE_THRESHOLD = 0.05


def summarise_cycle(model: SeaIceColumn, x_start: np.ndarray,
                    steps_per_year: int = 400,
                    store_every: int = 10) -> AnnualSummary:
    """Integrate one year from ``x_start`` and summarise the cycle."""
    t, x = model.annual_cycle(x_start, steps_per_year=steps_per_year,
                              store_every=store_every)
    h = model.thickness(x[:, 0])
    phi = model.open_fraction(x[:, 0])
    ice_free = h < ICE_FREE_THRESHOLD
    return AnnualSummary(
        thickness_mean=float(h.mean()),
        thickness_max=float(h.max()),
        thickness_min=float(h.min()),
        ice_free_fraction=float(ice_free.mean()),
        deep_temperature_mean=float(x[:, 1].mean()),
        open_fraction_mean=float(phi.mean()),
        perennial=bool(h.min() >= ICE_FREE_THRESHOLD),
    )
