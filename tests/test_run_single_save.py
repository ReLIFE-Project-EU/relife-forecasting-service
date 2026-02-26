from pathlib import Path

from fastapi.testclient import TestClient

import relife_forecasting.main as main_module


client = TestClient(main_module.app)


def test_run_single_save_with_archetype_from_building_examples(monkeypatch, tmp_path):
    """Run baseline-only single-save flow using an archetype declared in building_examples.py."""

    calls = {}

    def _fake_iso_calc(
        bui,
        weather_source="pvgis",
        sankey_graph=False,
        path_weather_file=None,
    ):
        calls["building_name"] = bui.get("building", {}).get("name")
        calls["weather_source"] = weather_source
        calls["sankey_graph"] = sankey_graph
        calls["path_weather_file"] = path_weather_file

        hourly = [
            {"Q_H_ideal": 1200.0, "Q_C_ideal": 150.0},
            {"Q_H_ideal": 1100.0, "Q_C_ideal": 120.0},
        ]
        annual = [{"Q_H_annual": 2300.0, "Q_C_annual": 270.0}]
        return hourly, annual

    monkeypatch.setattr(
        main_module.pybui.ISO52016,
        "Temperature_and_Energy_needs_calculation",
        _fake_iso_calc,
    )

    output_dir = tmp_path / "ecm_single_out"
    bui_dir = tmp_path / "ecm_single_bui"

    response = client.post(
        "/ecm_application/run_single_save",
        params={
            "archetype": "true",
            "category": "Single Family House",
            "country": "Austria",
            "name": "SFH_1946_1969",
            "weather_source": "pvgis",
            "include_baseline": "true",
            "save_bui": "true",
            "output_dir": str(output_dir),
            "bui_dir": str(bui_dir),
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "completed"
    assert payload["building"]["name"] == "SFH_1946_1969"
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["successful"] == 1
    assert payload["summary"]["failed"] == 0

    result = payload["results"][0]
    assert result["status"] == "success"
    assert result["scenario_id"] == "baseline"
    assert result["combo"] == []
    assert result["integrated_results"] is None

    files = result["files"]
    assert Path(files["hourly_csv"]).exists()
    assert Path(files["annual_csv"]).exists()
    assert Path(files["bui_json"]).exists()

    assert calls["building_name"] == "SFH_1946_1969"
    assert calls["weather_source"] == "pvgis"
    assert calls["sankey_graph"] is False
    assert calls["path_weather_file"] is None
