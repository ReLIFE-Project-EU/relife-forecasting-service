import json

import pandas as pd
from fastapi.testclient import TestClient

import relife_forecasting.main as main_module


client = TestClient(main_module.app)


def test_custom_ecm_uses_supplied_archetype_uni_config(monkeypatch):
    archetype = next(
        entry
        for entry in main_module.BUILDING_ARCHETYPES
        if entry["name"] == "GR_SFH_2011-now"
    )

    def fake_iso_simulation(bui, **kwargs):
        _ = bui, kwargs
        hourly = pd.DataFrame({"Q_H": [1000.0], "Q_C": [1000.0]})
        return hourly, pd.DataFrame({"Q_H_annual": [1000.0]})

    monkeypatch.setattr(main_module, "run_iso52016_simulation", fake_iso_simulation)

    params = {"baseline_only": "true", "weather_source": "pvgis"}
    reference = client.post(
        "/ecm_application",
        params={
            **params,
            "archetype": "true",
            "category": archetype["category"],
            "country": archetype["country"],
            "name": archetype["name"],
        },
    )
    custom = client.post(
        "/ecm_application",
        params={**params, "archetype": "false"},
        data={
            "bui_json": json.dumps(archetype["bui"]),
            "uni11300_json": json.dumps(archetype["uni11300"]),
        },
    )
    fallback = client.post(
        "/ecm_application",
        params={**params, "archetype": "false"},
        data={"bui_json": json.dumps(archetype["bui"])},
    )

    assert reference.status_code == custom.status_code == fallback.status_code == 200
    summary = lambda response: response.json()["scenarios"][0]["results"][
        "primary_energy_uni11300"
    ]["summary"]
    assert summary(custom) == summary(reference)
    assert summary(fallback) != summary(reference)
    assert custom.json()["uni11300_config_source"] == "custom"
    assert fallback.json()["uni11300_config_source"] == "example"


def test_custom_uni_config_validation_rejects_invalid_payloads():
    archetype = next(
        entry
        for entry in main_module.BUILDING_ARCHETYPES
        if entry["name"] == "GR_SFH_2011-now"
    )
    params = {"archetype": "false", "baseline_only": "true"}
    for value in ("not-json", "[]", "{}", '{"input_unit":"kWh"}'):
        response = client.post(
            "/ecm_application",
            params=params,
            data={"bui_json": json.dumps(archetype["bui"]), "uni11300_json": value},
        )
        assert response.status_code == 400

    response = client.post(
        "/ecm_application",
        params={
            "archetype": "true",
            "category": archetype["category"],
            "country": archetype["country"],
            "name": archetype["name"],
        },
        data={"uni11300_json": json.dumps(archetype["uni11300"])},
    )
    assert response.status_code == 400
