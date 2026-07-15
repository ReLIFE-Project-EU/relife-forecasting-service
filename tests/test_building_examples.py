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


def test_spec_provided_net_floor_area_is_preserved():
    # Legacy archetypes ship real areas in their specs and must pass through
    # unchanged (SFH_0_1945 declares 125.0).
    archetype = next(
        entry for entry in BUILDING_ARCHETYPES if entry["name"] == "SFH_0_1945"
    )

    assert archetype["bui"]["building"]["net_floor_area"] == 125.0


if __name__ == "__main__":
    test_all_archetypes_have_positive_net_floor_area()
    test_derived_net_floor_area_matches_footprint_times_floors()
    test_spec_provided_net_floor_area_is_preserved()
