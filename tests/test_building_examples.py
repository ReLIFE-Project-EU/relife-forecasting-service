import subprocess
import sys
from pathlib import Path

import pytest

from relife_forecasting import building_examples
from relife_forecasting.building_examples import BUILDING_ARCHETYPES
from relife_forecasting.catalogue.builders import build_archetype
from relife_forecasting.catalogue.spec import (
    ArchetypeSpec,
    BuildingCategory,
    BuildingTypeClass,
    ConstructionClass,
    Country,
)

SAMPLE_ARCHETYPE_NAME = "XX_AB_1970-1979"
CARDINAL_AZIMUTHS = [0, 90, 180, 270]


def test_builtin_archetype_inventory():
    countries = [
        "AT",
        "BE",
        "BG",
        "HR",
        "CY",
        "CZ",
        "DK",
        "EE",
        "FI",
        "FR",
        "DE",
        "GR",
        "HU",
        "IE",
        "IT",
        "LV",
        "LT",
        "LU",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SK",
        "SI",
        "ES",
        "SE",
    ]

    periods = [
        "0-1945",
        "1946-1969",
        "1970-1979",
        "1980-1989",
        "1990-1999",
        "2000-2010",
        "2011-now",
    ]

    expected = {
        f"{country}_{kind}_{period}"
        for country in countries
        for kind in ("SFH", "MFH", "AB")
        for period in periods
    }

    names = [entry["name"] for entry in BUILDING_ARCHETYPES]
    assert set(names) == expected
    assert len(names) == len(expected)


def test_single_run_script_can_list_archetypes_without_service_dependencies(tmp_path):
    """Users can browse the catalogue before installing the simulation dependencies."""
    script = Path(__file__).with_name("run_single_save_main.py").resolve()
    # Start outside the checkout with PYTHONPATH and installed packages disabled.
    result = subprocess.run(
        [sys.executable, "-I", "-S", script, "--list-archetypes"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert "AT_SFH_1946-1969" in result.stdout


def test_all_archetypes_have_positive_net_floor_area():
    zero_area = [
        archetype["name"]
        for archetype in BUILDING_ARCHETYPES
        if archetype["bui"]["building"]["net_floor_area"] <= 0
    ]

    assert zero_area == []


def sample_spec(**overrides) -> ArchetypeSpec:
    spec: ArchetypeSpec = {
        "country": Country.AUSTRIA.value,
        "category": BuildingCategory.APARTMENT_BUILDINGS.value,
        "latitude": 48.0,
        "longitude": 16.0,
        "exposed_perimeter": 30.0,
        "wall_thickness": 0.3,
        "n_floors": 1.5,
        "building_type_class": BuildingTypeClass.APARTMENT.value,
        "construction_class": ConstructionClass.I.value,
        "net_floor_area": 0.0,
        "roof_area": 40.0,
        "slab_area": 40.0,
        "wall_area": 100.0,
        "window_area": 20.0,
        "roof_u": 1.0,
        "slab_u": 1.0,
        "wall_u": 1.0,
        "window_u": 2.0,
        "window_height": 2.0,
        "window_width": 10.0,
        "roof_thermal_capacity": 100.0,
        "slab_thermal_capacity": 200.0,
        "wall_thermal_capacity": 300.0,
        "heating_eta_generation": 0.8,
        "cooling_cop_generation": 3.0,
        "dhw_eta_generation": 0.9,
    }
    spec.update(overrides)
    return spec


@pytest.mark.parametrize("reported_area", [0.0, -1.0], ids=["zero", "negative"])
def test_nonpositive_floor_area_is_derived_from_footprint_and_storeys(reported_area):
    spec = sample_spec(net_floor_area=reported_area, slab_area=40.0, n_floors=1.5)
    building = build_archetype(SAMPLE_ARCHETYPE_NAME, spec)["bui"]["building"]

    assert building["net_floor_area"] == spec["slab_area"] * spec["n_floors"]


def test_positive_floor_area_is_used_as_reported():
    spec = sample_spec(net_floor_area=45.0, slab_area=40.0, n_floors=1.5)
    building = build_archetype(SAMPLE_ARCHETYPE_NAME, spec)["bui"]["building"]

    assert building["net_floor_area"] == spec["net_floor_area"]


def test_height_and_volume_follow_the_footprint_and_storeys():
    spec = sample_spec(slab_area=40.0, n_floors=1.5, net_floor_area=45.0)
    building = build_archetype(SAMPLE_ARCHETYPE_NAME, spec)["bui"]["building"]

    storey_height_m = 2.7
    expected_height_m = spec["n_floors"] * storey_height_m
    assert building["height"] == pytest.approx(expected_height_m)
    assert building["volume"] == pytest.approx(spec["slab_area"] * expected_height_m)


@pytest.mark.parametrize("reported_area", [0.0, -1.0, 45.0], ids=["zero", "negative", "positive"])
def test_cooling_capacity_uses_the_resolved_net_floor_area(reported_area):
    spec = sample_spec(net_floor_area=reported_area)
    bui = build_archetype(SAMPLE_ARCHETYPE_NAME, spec)["bui"]
    cooling_capacity_w = bui["building_parameters"]["system_capacities"]["cooling_capacity"]

    cooling_load_w_per_m2 = 100
    assert cooling_capacity_w == pytest.approx(
        bui["building"]["net_floor_area"] * cooling_load_w_per_m2
    )


def test_each_facade_gets_an_equal_share_of_the_opaque_wall_area():
    spec = sample_spec(wall_area=100.0, window_area=20.0)
    surfaces = build_archetype(SAMPLE_ARCHETYPE_NAME, spec)["bui"]["building_surface"]
    walls = [
        surface
        for surface in surfaces
        if surface["type"] == "opaque" and surface["orientation"]["tilt"] == 90
    ]

    assert sorted(wall["orientation"]["azimuth"] for wall in walls) == CARDINAL_AZIMUTHS
    # The workbook's gross wall area includes the window openings.
    opaque_wall_area = spec["wall_area"] - spec["window_area"]
    for wall in walls:
        assert wall["area"] == pytest.approx(opaque_wall_area / len(CARDINAL_AZIMUTHS))


def test_window_area_and_width_are_shared_equally_across_facades():
    spec = sample_spec(window_area=20.0, window_width=10.0)
    surfaces = build_archetype(SAMPLE_ARCHETYPE_NAME, spec)["bui"]["building_surface"]
    windows = [surface for surface in surfaces if surface["type"] == "transparent"]

    assert sorted(window["orientation"]["azimuth"] for window in windows) == CARDINAL_AZIMUTHS
    for window in windows:
        assert window["area"] == pytest.approx(spec["window_area"] / len(CARDINAL_AZIMUTHS))
        assert window["width"] == pytest.approx(spec["window_width"] / len(CARDINAL_AZIMUTHS))


def test_windows_default_to_76_percent_solar_transmittance():
    surfaces = build_archetype(SAMPLE_ARCHETYPE_NAME, sample_spec())["bui"]["building_surface"]
    transmittances = {s["g_value"] for s in surfaces if s["type"] == "transparent"}
    assert transmittances == {0.76}


def test_building_infiltration_and_uni_ventilation_use_their_own_rates():
    spec = sample_spec(infiltration_rate=0.4, ventilation_rate_1_per_h=0.6)
    entry = build_archetype(SAMPLE_ARCHETYPE_NAME, spec)
    building_airflow = entry["bui"]["building_parameters"]["airflow_rates"]
    uni_ventilation = entry["uni11300"]["ventilation_params"]

    assert building_airflow["infiltration_rate"] == spec["infiltration_rate"]
    assert uni_ventilation["design_air_change_rate_1_per_h"] == spec["ventilation_rate_1_per_h"]


def test_internal_gains_use_the_reported_occupant_and_appliance_loads():
    spec = sample_spec(occupants_full_load=2.0, appliances_full_load=4.0)
    bui = build_archetype(SAMPLE_ARCHETYPE_NAME, spec)["bui"]
    gains = bui["building_parameters"]["internal_gains"]
    full_load_by_source = {gain["name"]: gain["full_load"] for gain in gains}

    assert full_load_by_source["occupants"] == spec["occupants_full_load"]
    assert full_load_by_source["appliances"] == spec["appliances_full_load"]


def test_thermal_bridge_heat_loss_uses_the_reported_delta_u_over_the_gross_envelope():
    spec = sample_spec(thermal_bridge_delta_u=0.2, roof_area=45.0, wall_area=100.0, slab_area=40.0)
    bui = build_archetype(SAMPLE_ARCHETYPE_NAME, spec)["bui"]
    heat_loss_w_per_k = bui["building_parameters"]["construction"]["thermal_bridge_heat_W_K"]

    gross_envelope_area = spec["roof_area"] + spec["wall_area"] + spec["slab_area"]
    assert heat_loss_w_per_k == pytest.approx(spec["thermal_bridge_delta_u"] * gross_envelope_area)


def test_named_exports_reference_catalogue_payloads():
    for entry in BUILDING_ARCHETYPES:
        assert getattr(building_examples, entry["name"] + "_BUILDING") is entry["bui"]
        assert getattr(building_examples, entry["name"] + "_UNI11300") is entry["uni11300"]
        assert entry["systems_archetype"] is entry["uni11300"]
        assert entry["system"] is building_examples.SYSTEM_CONDENSING_BOILER_RADIATORS


@pytest.mark.parametrize(
    "archetype",
    [entry for entry in BUILDING_ARCHETYPES if entry["country"] == Country.BELGIUM],
    ids=lambda entry: entry["name"],
)
def test_belgian_floor_area_matches_vertical_envelope(archetype):
    bui = archetype["bui"]
    building = bui["building"]
    vertical_area = sum(
        surface["area"]
        for surface in bui["building_surface"]
        if surface["orientation"]["tilt"] == 90
    )
    footprint = next(
        surface["area"] for surface in bui["building_surface"] if surface["name"].startswith("Slab")
    )
    inferred_floors = vertical_area / (building["exposed_perimeter"] * 2.7)
    assert building["net_floor_area"] == pytest.approx(
        footprint * inferred_floors, rel=0.002
    )  # source envelope dimensions are rounded
