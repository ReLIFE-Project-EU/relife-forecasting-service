import json

import pandas as pd
from fastapi.testclient import TestClient

import relife_forecasting.main as main_module


client = TestClient(main_module.app)


def test_ecm_lean_trims_hourly_output_only(monkeypatch):
    archetype = next(
        entry
        for entry in main_module.BUILDING_ARCHETYPES
        if entry["name"] == "GR_SFH_2011-now"
    )

    def fake_iso_simulation(bui, **kwargs):
        _ = bui, kwargs
        hourly = pd.DataFrame(
            {
                "Q_H": [1000.0],
                "Q_C": [500.0],
                "Q_HC": [500.0],
                "T_op": [21.0],
                "T_ext": [5.0],
                "Q_ve": [100.0],
            }
        )
        return hourly, pd.DataFrame({"Q_H_annual": [1000.0]})

    monkeypatch.setattr(main_module, "run_iso52016_simulation", fake_iso_simulation)

    def post(**params):
        response = client.post(
            "/ecm_application",
            params={"archetype": "false", "baseline_only": "true", **params},
            data={"bui_json": json.dumps(archetype["bui"])},
        )
        assert response.status_code == 200, response.text
        return response.json()["scenarios"][0]["results"]

    full = post()
    lean = post(lean="true")

    assert set(lean["hourly_building"]) == set(main_module.LEAN_HOURLY_COLUMNS)
    assert "hourly_results" in full["primary_energy_uni11300"]
    assert "hourly_results" not in lean["primary_energy_uni11300"]
    assert lean["primary_energy_uni11300"]["summary"] == full["primary_energy_uni11300"]["summary"]
    assert lean["annual_building"] == full["annual_building"]
