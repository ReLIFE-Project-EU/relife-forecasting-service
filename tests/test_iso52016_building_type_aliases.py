import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from relife_forecasting.main import app


client = TestClient(app)


def _minimal_bui(building_type_class: str) -> dict:
    return {
        "building": {
            "name": "Custom building",
            "building_type_class": building_type_class,
        },
        "building_surface": [
            {
                "name": "Opaque south surface",
                "type": "opaque",
                "area": 10.0,
                "sky_view_factor": 0.5,
                "u_value": 1.8,
                "orientation": {"tilt": 90, "azimuth": 180},
            }
        ],
        "building_parameters": {},
    }


@pytest.mark.parametrize(
    ("original_bt", "expected_bt"),
    [
        ("Residential_single_family", "Residential_detached_house"),
        ("Residential_multi_family", "Residential_apartment"),
    ],
)
def test_ecm_application_normalizes_building_type_aliases(monkeypatch, original_bt, expected_bt):
    captured: dict = {}

    def fake_iso52016(bui, weather_source="pvgis", sankey_graph=False, path_weather_file=None):
        captured["building_type_class"] = bui["building"]["building_type_class"]
        return (
            pd.DataFrame({"Q_H": [0.0], "Q_C": [0.0]}),
            pd.DataFrame({"annual": [0.0]}),
        )

    monkeypatch.setattr(
        "relife_forecasting.routes.forecasting_service_functions.pybui.ISO52016.Temperature_and_Energy_needs_calculation",
        fake_iso52016,
    )

    response = client.post(
        "/ecm_application",
        params={
            "archetype": "false",
            "weather_source": "pvgis",
            "u_wall": 0.25,
        },
        data={"bui_json": json.dumps(_minimal_bui(original_bt))},
    )

    assert response.status_code == 200, response.text
    assert captured["building_type_class"] == expected_bt
    assert response.json()["n_scenarios"] == 1


def test_ecm_application_rejects_unsupported_building_type_class(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("ISO 52016 should not run for unsupported building_type_class values.")

    monkeypatch.setattr(
        "relife_forecasting.routes.forecasting_service_functions.pybui.ISO52016.Temperature_and_Energy_needs_calculation",
        fail_if_called,
    )

    response = client.post(
        "/ecm_application",
        params={
            "archetype": "false",
            "weather_source": "pvgis",
            "u_wall": 0.25,
        },
        data={"bui_json": json.dumps(_minimal_bui("Warehouse"))},
    )

    assert response.status_code == 422
    assert "Unsupported building.building_type_class='Warehouse'" in response.text
