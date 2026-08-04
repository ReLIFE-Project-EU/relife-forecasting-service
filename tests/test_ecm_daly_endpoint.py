from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

import relife_forecasting.main as main_module


client = TestClient(main_module.app)


def _hourly_temperature_profile(value: float, days: int = 2) -> list[dict]:
    return [
        {"T_op": value, "Q_H": 0.0, "Q_C": 0.0}
        for _ in range(days * 24)
    ]


def test_ecm_daly_compares_baseline_and_intervention(monkeypatch) -> None:
    captured = {}

    async def fake_simulate_uvalues(**kwargs):
        captured.update(kwargs)
        return {
            "source": "custom",
            "name": "Test building",
            "category": "Residential",
            "country": "Italy",
            "weather_source": "pvgis",
            "scenarios": [
                {
                    "scenario_id": "baseline",
                    "description": "Baseline",
                    "results": {"hourly_building": _hourly_temperature_profile(19.0)},
                },
                {
                    "scenario_id": "wall",
                    "description": "Wall insulation",
                    "elements": ["wall"],
                    "u_values": {"wall": 0.3},
                    "results": {"hourly_building": _hourly_temperature_profile(22.0)},
                },
            ],
        }

    monkeypatch.setattr(main_module, "simulate_uvalues", fake_simulate_uvalues)

    response = client.post(
        "/ecm_application/daly",
        params={
            "archetype": "false",
            "u_wall": 0.3,
            "assessment_days": 2,
            "exposure_days_for_period": 365,
            "population_persons": 100000,
            "selected_threshold_pair": "COMFORT_PAIR_26_20",
        },
        data={"bui_json": "{}"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert captured["include_baseline"] is True
    assert captured["u_wall"] == 0.3
    assert payload["method"]["temperature_metric"] == "daily mean T_op from ISO 52016"
    assert payload["baseline"]["daly"]["valid_daily_temperature_rows"] == 2
    assert payload["baseline"]["daily_mean_internal_temperature_C"] == pytest.approx([19.0, 19.0])

    scenario = payload["ecm_scenarios"][0]
    assert scenario["daily_mean_internal_temperature_C"] == pytest.approx([22.0, 22.0])
    assert scenario["comparison_vs_baseline"]["cold"]["avoided_DALY"] > 0
    assert scenario["comparison_vs_baseline"]["total"]["reduction_percent"] == pytest.approx(100.0)
    assert scenario["comparison_vs_baseline"]["interpretation"] == "health_cobenefit"
    assert payload["ranking_by_avoided_total_DALY"][0]["scenario_id"] == "wall"


def test_ecm_daly_rejects_missing_temperature_column(monkeypatch) -> None:
    async def fake_simulate_uvalues(**kwargs):
        _ = kwargs
        hourly = [{"Q_H": 0.0, "Q_C": 0.0} for _ in range(24)]
        return {
            "scenarios": [
                {"scenario_id": "baseline", "results": {"hourly_building": hourly}},
                {"scenario_id": "wall", "results": {"hourly_building": hourly}},
            ]
        }

    monkeypatch.setattr(main_module, "simulate_uvalues", fake_simulate_uvalues)
    response = client.post(
        "/ecm_application/daly",
        params={"archetype": "false", "u_wall": 0.3, "assessment_days": 1},
        data={"bui_json": "{}"},
    )

    assert response.status_code == 422
    assert "T_op" in response.json()["detail"]
