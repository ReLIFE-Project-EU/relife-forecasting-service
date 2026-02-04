from pathlib import Path

import pytest
import requests


# TODO: Make this test self-contained by:
# 1. Using FastAPI TestClient instead of requiring a running server at localhost:9091
# 2. Either embedding a minimal EPW fixture or mocking the weather file dependency
@pytest.mark.skip(
    reason="Not self-contained: requires external EPW file and running server"
)
def test_simulate_using_api():
    """Test the /simulate API endpoint (PVGIS) and UNI/TS 11300 integration.

    Example response excerpt:
        {
          "results": {
            "primary_energy_uni11300": {
              "summary": { ... }
            }
          }
        }
    """

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

    resp = requests.post(url, params=params, headers=headers)

    resp.raise_for_status()
    payload = resp.json()
    assert payload
    assert "results" in payload
    assert "primary_energy_uni11300" in payload["results"]
    assert "summary" in payload["results"]["primary_energy_uni11300"]
