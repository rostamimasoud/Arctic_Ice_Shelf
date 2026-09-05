"""Tests of the column model and its forcing.

These check properties that must hold for the analysis to mean anything, rather
than fixed numbers that would merely record whatever the code did last.
"""

from __future__ import annotations

import numpy as np
import pytest

from arcticgnn.config import LATENT_ICE
from arcticgnn.physics.insolation import daily_insolation, surface_shortwave
from arcticgnn.physics.seaice import SeaIceColumn, SeaIceParams, summarise_cycle


# --------------------------------------------------------------------------- #
# Insolation
# --------------------------------------------------------------------------- #

def test_polar_night_and_polar_day():
    """The pole receives nothing in midwinter and a great deal in midsummer."""
    times = np.array([0.0, 0.5])
    values = daily_insolation(np.array([89.5]), times)[:, 0]
    assert values[0] == pytest.approx(0.0, abs=1e-6)
    assert values[1] > 400.0


def test_insolation_annual_mean_decreases_polewards():
    times = np.linspace(0.0, 1.0, 365, endpoint=False)
    means = daily_insolation(np.array([50.0, 70.0, 89.0]), times).mean(axis=0)
    assert means[0] > means[1] > means[2]


def test_insolation_is_never_negative():
    times = np.linspace(0.0, 1.0, 365, endpoint=False)
    values = surface_shortwave(np.arange(55.0, 90.0, 1.0), times)
    assert values.min() >= 0.0


# --------------------------------------------------------------------------- #
# State mapping
# --------------------------------------------------------------------------- #

def test_thickness_round_trip():
    model = SeaIceColumn()
    for thickness in (0.1, 1.0, 3.5):
        enthalpy = -thickness * LATENT_ICE
        assert model.thickness(enthalpy) == pytest.approx(thickness)


def test_no_ice_when_enthalpy_positive():
    model = SeaIceColumn()
    assert model.thickness(5.0) == 0.0


def test_open_fraction_spans_zero_to_one():
    model = SeaIceColumn()
    assert model.open_fraction(-40.0) < 1e-3
    assert model.open_fraction(40.0) > 1.0 - 1e-3


def test_albedo_lies_between_its_limits():
    params = SeaIceParams()
    model = SeaIceColumn(params)
    values = model.albedo(np.linspace(-60.0, 60.0, 101))
    assert values.min() >= min(params.albedo_ice, params.albedo_ocean) - 1e-9
    assert values.max() <= max(params.albedo_ice, params.albedo_ocean) + 1e-9


def test_ice_surface_never_rises_above_melting_point():
    """Energy beyond the melting point must melt ice and not warm the surface."""
    model = SeaIceColumn()
    enthalpy = np.linspace(-40.0, -0.5, 60)
    for time in np.linspace(0.0, 1.0, 13):
        temperature = model.surface_temperature(enthalpy, float(time))
        assert temperature.max() <= 1e-6


# --------------------------------------------------------------------------- #
# Smoothness, which the continuation depends on
# --------------------------------------------------------------------------- #

def test_tendency_is_smooth_across_the_freezing_point():
    """No kink where ice gives way to open water.

    The continuation differentiates a one year integration by finite
    differences, so a discontinuous derivative here would make every Jacobian
    near the threshold meaningless.
    """
    model = SeaIceColumn()
    enthalpy = np.linspace(-3.0, 3.0, 601)
    state = np.stack([enthalpy, np.full_like(enthalpy, 1.5)], axis=-1)
    tendency = model.rhs(0.3, state)[:, 0]

    second = np.diff(tendency, 2)
    # A kink would leave one isolated spike far above the typical curvature.
    assert np.abs(second).max() < 40.0 * np.median(np.abs(second)) + 1e-9


# --------------------------------------------------------------------------- #
# Integration and attractors
# --------------------------------------------------------------------------- #

def test_annual_map_fixes_the_spun_up_state():
    """A settled state must be a fixed point of the one year map."""
    model = SeaIceColumn(SeaIceParams().replace(longwave_offset=74.6, albedo_scale=0.5))
    settled = model.spin_up(model.initial_state("perennial"), years=300.0,
                            steps_per_year=200)
    advanced = model.annual_map(settled, steps_per_year=200)
    assert np.allclose(advanced, settled, atol=2e-3)


def test_integration_is_deterministic_without_noise():
    """Required for finite difference Jacobians to be meaningful."""
    model = SeaIceColumn()
    start = model.initial_state("perennial")
    first = model.integrate(start, years=5.0, steps_per_year=200)[1][-1]
    second = model.integrate(start, years=5.0, steps_per_year=200)[1][-1]
    assert np.array_equal(first, second)


def test_batched_integration_matches_single_columns():
    """A batch must give exactly what the columns give one at a time."""
    model = SeaIceColumn()
    a = np.array([-2.0 * LATENT_ICE, 0.8])
    b = np.array([-0.5 * LATENT_ICE, 2.0])
    together = model.integrate(np.stack([a, b]), years=3.0,
                               steps_per_year=200)[1][-1]
    apart = np.stack([model.integrate(a, years=3.0, steps_per_year=200)[1][-1],
                      model.integrate(b, years=3.0, steps_per_year=200)[1][-1]])
    assert np.allclose(together, apart, atol=1e-10)


def test_calibrated_column_keeps_ice_through_the_year():
    model = SeaIceColumn(SeaIceParams().replace(longwave_offset=74.6, albedo_scale=0.5))
    settled = model.spin_up(model.initial_state("perennial"), years=300.0,
                            steps_per_year=200)
    summary = summarise_cycle(model, settled, steps_per_year=200)
    assert summary.perennial
    assert 1.0 < summary.thickness_mean < 2.5
    assert summary.thickness_max > summary.thickness_min


def test_warming_thins_the_ice():
    """Monotone response to the control parameter on the perennial branch."""
    base = SeaIceParams().replace(longwave_offset=74.6, albedo_scale=0.5)
    thicknesses = []
    for forcing in (0.0, 1.0, 2.0):
        model = SeaIceColumn(base.replace(forcing=forcing))
        settled = model.spin_up(model.initial_state("perennial"), years=250.0,
                                steps_per_year=200)
        thicknesses.append(
            summarise_cycle(model, settled, steps_per_year=200).thickness_mean)
    assert thicknesses[0] > thicknesses[1] > thicknesses[2]


def test_stored_heat_rises_under_a_closed_cover():
    """The subsurface feedback must actually trap heat when the ice is thick."""
    params = SeaIceParams().replace(longwave_offset=74.6, albedo_scale=0.5)
    model = SeaIceColumn(params)
    thick = model.spin_up(np.array([-4.0 * LATENT_ICE, 0.2]), years=300.0,
                          steps_per_year=200)
    thin = model.spin_up(np.array([2.0 * params.c_mixed, 0.2]), years=300.0,
                         steps_per_year=200)
    # Under thick ice mixing is weak, so the interior warms towards the inflow.
    assert thick[1] > thin[1]


def test_unknown_control_parameter_is_rejected():
    with pytest.raises(ValueError):
        SeaIceParams().with_control(nonexistent=1.0)


def test_unknown_regime_is_rejected():
    with pytest.raises(ValueError):
        SeaIceColumn().initial_state("tropical")
