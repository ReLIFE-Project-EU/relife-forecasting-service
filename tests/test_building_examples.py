import copy
import json
import os
import subprocess
import sys

import pytest

from relife_forecasting.building_examples import (
    BUILDING_ARCHETYPES,
    _EXTENDED_ARCHETYPE_SPECS,
    _apply_provisional_thermal_mass,
)


def test_all_archetypes_have_positive_net_floor_area():
    zero_area = [
        archetype["name"]
        for archetype in BUILDING_ARCHETYPES
        if archetype["bui"]["building"]["net_floor_area"] <= 0
    ]

    assert zero_area == []


def test_derived_net_floor_area_matches_footprint_times_floors():
    # AT_SFH_0-1945 ships a placeholder 0.0 in its spec, so its area must be
    # derived from the ground slab (74.9 m2) times the floor count (2).
    archetype = next(
        entry for entry in BUILDING_ARCHETYPES if entry["name"] == "AT_SFH_0-1945"
    )

    assert archetype["bui"]["building"]["net_floor_area"] == 74.9 * 2


@pytest.mark.parametrize(
    "archetype",
    [entry for entry in BUILDING_ARCHETYPES if entry["country"] == "Belgium"],
    ids=lambda entry: entry["name"],
)
def test_belgian_floor_area_matches_vertical_envelope(archetype):
    bui = archetype["bui"]
    building = bui["building"]
    vertical_area = sum(
        surface["area"] for surface in bui["building_surface"]
        if surface["orientation"]["tilt"] == 90
    )
    footprint = next(
        surface["area"] for surface in bui["building_surface"]
        if surface["name"].startswith("Slab")
    )
    inferred_floors = vertical_area / (building["exposed_perimeter"] * 2.7)
    assert building["net_floor_area"] == pytest.approx(
        footprint * inferred_floors, rel=0.002
    )  # source envelope dimensions are rounded


def test_unknown_thermal_mass_mode_is_rejected():
    archetype = BUILDING_ARCHETYPES[0]
    spec = _EXTENDED_ARCHETYPE_SPECS[archetype["name"]]
    with pytest.raises(ValueError, match="Unknown extended thermal-mass mode"):
        _apply_provisional_thermal_mass(archetype["bui"], spec, mode="unknown")


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("AT_SFH_0-1945", "class_i"),
        ("AT_SFH_1946-1969", "class_d"),
        ("AT_SFH_1970-1979", "class_d"),
    ],
)
def test_provisional_thermal_mass_converts_only_class_capacities_and_units(
    name, expected_class
):
    bui = next(entry["bui"] for entry in BUILDING_ARCHETYPES if entry["name"] == name)
    spec = _EXTENDED_ARCHETYPE_SPECS[name]
    original_bui, original_spec = copy.deepcopy(bui), copy.deepcopy(spec)

    converted = _apply_provisional_thermal_mass(bui, spec, mode="provisional")

    expected = copy.deepcopy(bui)
    expected["building"]["construction_class"] = expected_class
    expected["units"]["thermal_capacity"] = "J/(m²·K)"
    # One roof, four opaque wall sides, and one ground slab. Wall capacity is
    # divided by the combined gross wall area, not by a single side or net area.
    capacities = [
        spec["roof_thermal_capacity"] / spec["roof_area"],
        *[spec["wall_thermal_capacity"] / spec["wall_area"]] * 4,
        spec["slab_thermal_capacity"] / spec["slab_area"],
    ]
    for surface, capacity in zip(expected["building_surface"], capacities):
        surface["thermal_capacity"] = pytest.approx(capacity)
    assert converted == expected
    assert converted is not bui
    assert bui == original_bui
    assert spec == original_spec


def test_thermal_mass_modes_cover_the_current_catalogue():
    original = copy.deepcopy(BUILDING_ARCHETYPES)
    for archetype in BUILDING_ARCHETYPES:
        bui = archetype["bui"]
        spec = _EXTENDED_ARCHETYPE_SPECS[archetype["name"]]
        assert _apply_provisional_thermal_mass(bui, spec, mode="legacy") is bui
        converted = _apply_provisional_thermal_mass(bui, spec, mode="provisional")
        assert converted["building"]["construction_class"] in {"class_i", "class_d"}
        assert all(
            surface["thermal_capacity"] > 0
            for surface in converted["building_surface"]
            if surface["type"] == "opaque"
        ), archetype["name"]
    assert BUILDING_ARCHETYPES == original


@pytest.mark.parametrize("mode", [None, "legacy", "provisional", " Provisional ", "invalid"])
def test_thermal_mass_startup_configuration(mode):
    env = os.environ.copy()
    env.pop("RELIFE_EXTENDED_THERMAL_MASS_MODE", None)
    if mode is not None:
        env["RELIFE_EXTENDED_THERMAL_MASS_MODE"] = mode
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; "
            "from relife_forecasting.building_examples import BUILDING_ARCHETYPES; "
            "print(json.dumps(BUILDING_ARCHETYPES))",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if mode == "invalid":
        assert result.returncode != 0
        assert "RELIFE_EXTENDED_THERMAL_MASS_MODE must be" in result.stderr
        return

    assert result.returncode == 0, result.stderr
    catalogue = json.loads(result.stdout)
    assert len(catalogue) == len(_EXTENDED_ARCHETYPE_SPECS)
    provisional = mode is not None and mode.strip().lower() == "provisional"
    for archetype in catalogue:
        bui = archetype["bui"]
        spec = _EXTENDED_ARCHETYPE_SPECS[archetype["name"]]
        expected_class = spec["construction_class"]
        if provisional and expected_class != "class_i":
            expected_class = "class_d"
        assert bui["building"]["construction_class"] == expected_class
        assert bui["units"]["thermal_capacity"] == (
            "J/(m²·K)" if provisional else "J/K"
        )
        capacities = [
            spec["roof_thermal_capacity"],
            *[spec["wall_thermal_capacity"]] * 4,
            spec["slab_thermal_capacity"],
        ]
        areas = [spec["roof_area"], *[spec["wall_area"]] * 4, spec["slab_area"]]
        for surface, capacity, area in zip(bui["building_surface"], capacities, areas):
            assert surface["thermal_capacity"] == pytest.approx(
                capacity / area if provisional else capacity
            )


if __name__ == "__main__":
    test_all_archetypes_have_positive_net_floor_area()
    test_derived_net_floor_area_matches_footprint_times_floors()
