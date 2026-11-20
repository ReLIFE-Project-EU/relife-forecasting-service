import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from relife_forecasting.building_examples import BUILDING_ARCHETYPES
from relife_forecasting.catalogue.spec import BuildingCategory, Country
from relife_forecasting.routes.archetypes import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.mark.parametrize(
    "name,country,building_type,wall_area,window_area",
    [
        ("AT_SFH_1946-1969", Country.AUSTRIA, BuildingCategory.SINGLE_FAMILY_HOUSE, 176.7, 36.2),
        ("SE_AB_1990-1999", Country.SWEDEN, BuildingCategory.APARTMENT_BUILDINGS, 900.7, 327.9),
    ],
)
def test_capex_returns_reference_envelope_areas(
    name, country, building_type, wall_area, window_area
):
    filters = {"name": name, "country": country, "building_type": building_type}
    response = client.get("/building/archetypes_info_capex", params=filters)
    assert response.status_code == 200
    result = response.json()
    assert result["filters"] == filters
    assert result["count"] == len(result["archetypes"]) == 1
    summary = result["archetypes"][0]
    assert (summary["name"], summary["country"]) == (name, country)
    assert summary["surfaces"]["wall"]["area"] == pytest.approx(wall_area)
    assert summary["surfaces"]["window"]["area"] == pytest.approx(window_area)


@pytest.mark.parametrize(
    "country,building_type",
    [
        (Country.SWEDEN, BuildingCategory.SINGLE_FAMILY_HOUSE),
        (Country.AUSTRIA, BuildingCategory.APARTMENT_BUILDINGS),
    ],
)
def test_capex_excludes_named_archetype_when_another_filter_mismatches(country, building_type):
    response = client.get(
        "/building/archetypes_info_capex",
        params={"name": "AT_SFH_1946-1969", "country": country, "building_type": building_type},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["count"] == 0
    assert result["archetypes"] == []


def test_name_only_returns_grouped_matches():
    entry = BUILDING_ARCHETYPES[0]
    response = client.get(
        "/building/archetypes_info_capex", params={"name": "  " + entry["name"].lower()}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["countries"] == [entry["country"]]
    assert [item["name"] for item in payload["archetypes_by_country"][entry["country"]]] == [
        entry["name"]
    ]
