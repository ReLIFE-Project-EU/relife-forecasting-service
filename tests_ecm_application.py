import json

import requests


BASE_URL = "http://127.0.0.1:9091"


def test_ecm_application_custom_bui():
    """Test /ecm_application with a custom BUI against the running service."""

    bui_payload = {
        "building_surface": [
            {
                "type": "opaque",
                "orientation": {"tilt": 0, "azimuth": 0},
                "u_value": 1.5,
            },
            {
                "type": "opaque",
                "orientation": {"tilt": 90, "azimuth": 180},
                "u_value": 1.8,
            },
            {
                "type": "transparent",
                "orientation": {"tilt": 90, "azimuth": 180},
                "u_value": 2.7,
            },
        ]
    }

    response = requests.post(
        f"{BASE_URL}/ecm_application",
        params={
            "archetype": "false",
            "weather_source": "pvgis",
            "u_wall": 0.25,
        },
        data={"bui_json": json.dumps(bui_payload)},
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    assert data["source"] == "custom"
    assert data["weather_source"] == "pvgis"
    assert data["u_values_requested"]["wall"] == 0.25
    assert data["n_scenarios"] == 1
    assert len(data["scenarios"]) == 1

    scenario = data["scenarios"][0]
    assert scenario["scenario_id"] == "wall"
    assert "results" in scenario
    assert "annual_building" in scenario["results"]
    assert "hourly_building" in scenario["results"]


