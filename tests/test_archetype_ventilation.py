import copy
from pathlib import Path

import pytest

from relife_forecasting.building_examples import BUILDING_ARCHETYPES
from relife_forecasting.routes.forecasting_service_functions import (
    run_iso52016_simulation,
    scale_bui_floor_area,
)

EPW = Path(__file__).resolve().parents[1] / "src/relife_forecasting/epw_weather/GRC_Athens.167160_IWEC.epw"


def _arch(name):
    return next(a for a in BUILDING_ARCHETYPES if a["name"] == name)["bui"]


def test_every_archetype_declares_ventilation_and_volume():
    """pybuildingenergy >= 2.1 applies no ventilation without this block and a volume."""
    for arch in BUILDING_ARCHETYPES:
        bui = arch["bui"]
        comps = bui["building_parameters"]["ventilation"]["components"]
        assert len(comps) == 1, arch["name"]
        assert comps[0]["ventilation_type"] == "constant_ach", arch["name"]
        assert comps[0]["air_changes_per_hour"] == bui["building_parameters"]["airflow_rates"]["infiltration_rate"]
        assert bui["building"]["volume"] > 0, arch["name"]


def test_extended_volume_is_footprint_times_floors_times_storey_height():
    bui = _arch("AT_SFH_1946-1969")
    assert bui["building"]["volume"] == pytest.approx(82.7 * 2 * 2.7)


def test_scaling_floor_area_scales_volume():
    bui = _arch("AT_SFH_1946-1969")
    out = scale_bui_floor_area(bui, bui["building"]["net_floor_area"] * 1.5)
    assert out["building"]["volume"] == pytest.approx(bui["building"]["volume"] * 1.5)


def test_ventilation_actually_enters_the_simulation():
    bui = _arch("AT_SFH_1946-1969")
    hourly, _ = run_iso52016_simulation(
        copy.deepcopy(bui), weather_source="epw", path_weather_file=str(EPW)
    )
    # 0.8 ACH over 447 m3: H_ve ~ 0.34 * 0.8 * 447 ~ 120 W/K
    assert hourly["H_ve"].mean() > 100.0


def test_height_is_total_building_height_and_floor_area_is_per_building():
    for arch in BUILDING_ARCHETYPES:
        bd = arch["bui"]["building"]
        slab = next(s for s in arch["bui"]["building_surface"] if s["name"].startswith("Slab"))
        assert bd["height"] == pytest.approx(bd["n_floors"] * 2.7), arch["name"]
        assert bd["net_floor_area"] == pytest.approx(slab["area"] * bd["n_floors"]), arch["name"]
        assert bd["volume"] == pytest.approx(bd["net_floor_area"] * 2.7), arch["name"]


def test_air_change_and_thermal_bridges_follow_the_construction_period():
    periods = ["0-1945", "1946-1969", "1970-1979", "1980-1989", "1990-1999", "2000-2010", "2011-now"]
    ach, tb = [], []
    for p in periods:
        bui = _arch(f"AT_SFH_{p}")
        ach.append(bui["building_parameters"]["airflow_rates"]["infiltration_rate"])
        env = sum(
            s["area"] for s in bui["building_surface"]
            if s["name"].startswith(("Roof", "Slab", "Opaque")) or s["type"] == "transparent"
        )
        tb.append(bui["building_parameters"]["construction"]["thermal_bridge_heat_W_K"] / env)
    assert ach == sorted(ach, reverse=True) and 0.4 <= min(ach) and max(ach) <= 1.0
    assert tb == sorted(tb, reverse=True)
    assert tb[0] > tb[-1]


def test_internal_gains_and_profiles_follow_building_type():
    sfh = _arch("AT_SFH_1946-1969")["building_parameters"]
    apt = _arch("AT_MFH_1946-1969")["building_parameters"]
    loads = lambda bp: {g["name"]: g["full_load"] for g in bp["internal_gains"]}
    assert loads(sfh)["occupants"] == 2.8 and loads(apt)["occupants"] == 4.2
    assert loads(sfh)["appliances"] == 2.4 and loads(apt)["appliances"] == 3.0
    for bp in (sfh, apt):
        assert all(len(bp[k][d]) == 24 for k in ("heating_profile", "cooling_profile", "ventilation_profile") for d in ("weekday", "weekend"))
        occ = next(g for g in bp["internal_gains"] if g["name"] == "occupants")
        # Cooling only when people are at home: weekday daytime (low occupancy) is off.
        for hour, on in enumerate(bp["cooling_profile"]["weekday"]):
            if occ["weekday"][hour] <= 0.2:
                assert on == 0
        assert set(bp["ventilation_profile"]["weekday"]) == {1}


def test_capacities_follow_the_building_and_absorptance_is_uniform():
    for arch in BUILDING_ARCHETYPES:
        bui = arch["bui"]
        area = bui["building"]["net_floor_area"]
        caps = bui["building_parameters"]["system_capacities"]
        assert 20 <= caps["heating_capacity"] / area <= 250, arch["name"]
        assert caps["cooling_capacity"] == pytest.approx(100.0 * area), arch["name"]
        absorptance = {
            s["solar_absorptance"]
            for s in bui["building_surface"]
            if s["name"].startswith(("Roof", "Opaque"))
        }
        assert absorptance == {0.6}, arch["name"]


def test_heating_capacity_is_larger_in_colder_countries():
    cold = _arch("FI_SFH_2011-now")["building_parameters"]["system_capacities"]["heating_capacity"]
    mild = _arch("CY_SFH_2011-now")["building_parameters"]["system_capacities"]["heating_capacity"]
    assert cold / _arch("FI_SFH_2011-now")["building"]["net_floor_area"] > mild / _arch("CY_SFH_2011-now")["building"]["net_floor_area"]


def test_every_country_has_degree_days():
    from relife_forecasting.building_examples import _EXTENDED_ARCHETYPE_SPECS, _HDD_BY_COUNTRY

    countries = {s["country"] for s in _EXTENDED_ARCHETYPE_SPECS.values()}
    assert countries == set(_HDD_BY_COUNTRY)
    assert 2000 < _HDD_BY_COUNTRY["Belgium"] < 3000  # not the 1415 of the workbook
