from fastapi.testclient import TestClient
from fastapi import FastAPI

from relife_forecasting.routes.archetypes import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_archetypes_route_filters_by_country_type_and_name():
    response = client.get(
        "/building/archetypes",
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
    assert archetype["category"] == "Single Family House"


if __name__ == "__main__":
    test_archetypes_route_filters_by_country_type_and_name()