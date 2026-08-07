import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from relife_forecasting.building_examples import BUILDING_ARCHETYPES
from relife_forecasting.routes.archetypes import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_archetypes_route_filters_by_country_type_and_name():
    response = client.get(
        "/building/archetypes_info_capex",
        params={
            "country": "Austria",
            "building_type": "Single Family House",
            "name": "SFH_1946_1969",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["count"] == 1
    assert payload["filters"] == {
        "country": "Austria",
        "building_type": "Single Family House",
        "name": "SFH_1946_1969",
    }
    assert len(payload["archetypes"]) == 1
    archetype = payload["archetypes"][0]
    assert archetype["name"] == "SFH_1946_1969"
    assert archetype["country"] == "Austria"
    assert archetype["surfaces"]["wall"]["area"] == pytest.approx(208.8)
    assert archetype["surfaces"]["window"]["area"] == pytest.approx(39.6)


def test_generated_archetype_splits_surface_areas_across_sides():
    archetype = next(
        entry for entry in BUILDING_ARCHETYPES if entry["name"] == "SE_AB_1990-1999"
    )
    surfaces = archetype["bui"]["building_surface"]
    walls = [
        surface
        for surface in surfaces
        if surface["type"] == "opaque" and surface["orientation"]["tilt"] == 90
    ]
    windows = [
        surface
        for surface in surfaces
        if surface["type"] == "transparent" and surface["orientation"]["tilt"] == 90
    ]

    assert len(walls) == 4
    assert len(windows) == 4
    assert all(surface["area"] == pytest.approx(225.175) for surface in walls)
    assert all(surface["area"] == pytest.approx(81.975) for surface in windows)
    assert all(surface["width"] == pytest.approx(54.65) for surface in windows)

    response = client.get(
        "/building/archetypes_info_capex",
        params={
            "country": "Sweden",
            "building_type": "Apartment buildings",
            "name": "SE_AB_1990-1999",
        },
    )

    assert response.status_code == 200
    result = response.json()["archetypes"][0]
    assert result["surfaces"]["wall"]["area"] == pytest.approx(900.7)
    assert result["surfaces"]["window"]["area"] == pytest.approx(327.9)


if __name__ == "__main__":
    test_archetypes_route_filters_by_country_type_and_name()
