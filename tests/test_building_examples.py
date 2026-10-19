import pytest

from relife_forecasting.building_examples import BUILDING_ARCHETYPES


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


if __name__ == "__main__":
    test_all_archetypes_have_positive_net_floor_area()
    test_derived_net_floor_area_matches_footprint_times_floors()
