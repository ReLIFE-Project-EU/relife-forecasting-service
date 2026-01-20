from pathlib import Path

import requests


def test_simulate_using_api():
    """Test the /simulate API endpoint using a local EPW file."""

    url = "http://127.0.0.1:9091/simulate"

    params = {
        "archetype": "true",
        "category": "Single Family House",
        "country": "Greece",
        "name": "SFH_Greece_1946_1969",
        "weather_source": "pvgis",
    }

    headers = {
        "accept": "application/json",
    }

    epw_path = Path("epw_weather/2020_Athens.epw")

    data = {
        "bui_json": "string",
        "system_json": "string",
    }

    with epw_path.open("rb") as epw_file:
        files = {
            "epw_file": (epw_path.name, epw_file, "application/octet-stream"),
        }

        resp = requests.post(
            url, params=params, headers=headers, files=files, data=data
        )

    resp.raise_for_status()
    assert resp.json()
