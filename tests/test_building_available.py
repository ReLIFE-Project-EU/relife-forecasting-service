from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import relife_forecasting.main as main_module
from relife_forecasting import building_examples
from relife_forecasting.catalogue.spec import BuildingCategory, Country

client = TestClient(main_module.app)


def test_available_archetypes_preserves_identities_and_bui_coordinates():
    response = client.get("/building/available")

    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert len(payload) == len(main_module.BUILDING_ARCHETYPES)
    for actual, source in zip(payload, main_module.BUILDING_ARCHETYPES):
        building = source["bui"]["building"]
        assert actual == {
            "name": source["name"],
            "category": source["category"],
            "country": source["country"],
            "latitude": building.get("latitude"),
            "longitude": building.get("longitude"),
        }
        for coordinate in ("latitude", "longitude"):
            if actual[coordinate] is not None:
                assert type(actual[coordinate]) in (int, float)


@pytest.mark.parametrize(
    "bui, expected_latitude, expected_longitude",
    [
        ({}, None, None),
        ({"building": {}}, None, None),
        ({"building": {"latitude": 47.8095}}, 47.8095, None),
        ({"building": {"longitude": 13.055}}, None, 13.055),
        ({"building": {"latitude": None, "longitude": None}}, None, None),
        ({"building": {"latitude": 0, "longitude": 0.0}}, 0, 0.0),
    ],
)
def test_available_archetypes_serializes_optional_coordinates(
    monkeypatch, bui, expected_latitude, expected_longitude
):
    monkeypatch.setattr(
        main_module,
        "BUILDING_ARCHETYPES",
        [
            {
                "name": "Test reference",
                "category": "Single Family House",
                "country": "Austria",
                "bui": bui,
            }
        ],
    )

    response = client.get("/building/available")

    assert response.status_code == 200
    payload = response.json()
    assert payload == [
        {
            "name": "Test reference",
            "category": "Single Family House",
            "country": "Austria",
            "latitude": expected_latitude,
            "longitude": expected_longitude,
        }
    ]
    for coordinate in ("latitude", "longitude"):
        if payload[0][coordinate] is not None:
            assert type(payload[0][coordinate]) in (int, float)


def test_epc_archetype_is_visible_in_both_catalogue_routes():
    catalogue = building_examples.BUILDING_ARCHETYPES
    original = catalogue.copy()
    name = f"test-epc-{uuid4().hex}"
    try:
        created = client.post(
            "/bui/epc_update_u_values",
            params={
                "name": "AT_SFH_1946-1969",
                "category": BuildingCategory.SINGLE_FAMILY_HOUSE,
                "country": Country.AUSTRIA,
                "energy_class": "A",
                "new_name": name,
            },
        )
        assert created.status_code == 200, created.text
        assert created.json()["new_archetype_name"] == name

        available = client.get("/building/available")
        assert available.status_code == 200
        assert name in [entry["name"] for entry in available.json()]

        capex = client.get(
            "/building/archetypes_info_capex", params={"name": name, "country": Country.AUSTRIA}
        )
        assert capex.status_code == 200
        assert [entry["name"] for entry in capex.json()["archetypes"]] == [name]
    finally:
        catalogue[:] = original
