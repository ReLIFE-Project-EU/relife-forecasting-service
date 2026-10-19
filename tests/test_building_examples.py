import copy
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


@pytest.mark.parametrize(
    "mode,expected",
    [(None, "class_iii J/K"), ("provisional", "class_d J/(m²·K)"), ("invalid", None)],
)
def test_thermal_mass_startup_configuration(monkeypatch, mode, expected):
    monkeypatch.delenv("RELIFE_EXTENDED_THERMAL_MASS_MODE", raising=False)
    if mode is not None:
        monkeypatch.setenv("RELIFE_EXTENDED_THERMAL_MASS_MODE", mode)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from relife_forecasting.building_examples import BUILDING_ARCHETYPES; "
            "bui = next(a['bui'] for a in BUILDING_ARCHETYPES "
            "if a['name'] == 'AT_SFH_1946-1969'); "
            "print(bui['building']['construction_class'], bui['units']['thermal_capacity'])",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if expected is None:
        assert result.returncode != 0
        assert "RELIFE_EXTENDED_THERMAL_MASS_MODE must be" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected


if __name__ == "__main__":
    test_all_archetypes_have_positive_net_floor_area()
    test_derived_net_floor_area_matches_footprint_times_floors()
