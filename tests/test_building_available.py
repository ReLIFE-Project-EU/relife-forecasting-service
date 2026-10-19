from fastapi.testclient import TestClient
import pytest

import relife_forecasting.main as main_module


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
    monkeypatch.setattr(main_module, "BUILDING_ARCHETYPES", [{
        "name": "Test reference",
        "category": "Single Family House",
        "country": "Austria",
        "bui": bui,
    }])

    response = client.get("/building/available")

    assert response.status_code == 200
    payload = response.json()
    assert payload == [{
        "name": "Test reference",
        "category": "Single Family House",
        "country": "Austria",
        "latitude": expected_latitude,
        "longitude": expected_longitude,
    }]
    for coordinate in ("latitude", "longitude"):
        if payload[0][coordinate] is not None:
            assert type(payload[0][coordinate]) in (int, float)


@pytest.mark.parametrize(
    "country, retired_name, replacement_name",
    [
        ("Austria", "SFH_0_1945", "AT_SFH_0-1945"),
        ("Austria", "SFH_1946_1969", "AT_SFH_1946-1969"),
        ("Austria", "SFH_1970_1979", "AT_SFH_1970-1979"),
        ("Greece", "SFH_Greece_1946_1969", "GR_SFH_1946-1969"),
        ("Italy", "SFH_Italy_1946_1969", "IT_SFH_1946-1969"),
    ],
)
def test_retired_archetypes_are_removed_and_replacements_available(
    country, retired_name, replacement_name
):
    response = client.get("/building/available")
    assert response.status_code == 200
    identities = {
        (entry["country"], entry["category"], entry["name"])
        for entry in response.json()
    }
    category = "Single Family House"
    assert (country, category, retired_name) not in identities
    assert (country, category, replacement_name) in identities
    params = {"archetype": "true", "country": country, "category": category}
    assert client.post(
        "/building", params={**params, "name": retired_name}
    ).status_code == 404
    replacement = client.post(
        "/building", params={**params, "name": replacement_name}
    )
    assert replacement.status_code == 200
    assert replacement.json()["bui"]["building"]["name"] == replacement_name
