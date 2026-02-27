from pathlib import Path
from typing import Any, Dict

from fastapi.testclient import TestClient

import relife_forecasting.main as main_module


client = TestClient(main_module.app)


def _pick_test_archetype() -> Dict[str, str]:
    preferred = next(
        (
            arch
            for arch in main_module.BUILDING_ARCHETYPES
            if arch.get("category") == "Single Family House"
            and arch.get("country") == "Austria"
            and arch.get("name") == "SFH_1946_1969"
        ),
        None,
    )
    if preferred:
        return {
            "category": preferred["category"],
            "country": preferred["country"],
            "name": preferred["name"],
        }

    fallback = next(
        (
            arch
            for arch in main_module.BUILDING_ARCHETYPES
            if isinstance(arch.get("category"), str)
            and isinstance(arch.get("country"), str)
            and isinstance(arch.get("name"), str)
        ),
        None,
    )
    if fallback is None:
        raise AssertionError("No valid archetype found in BUILDING_ARCHETYPES.")

    return {
        "category": fallback["category"],
        "country": fallback["country"],
        "name": fallback["name"],
    }


def test_run_single_save_with_archetype_from_building_examples(monkeypatch, tmp_path):
    """Run baseline-only single-save flow using an archetype declared in building_examples.py."""

    calls: Dict[str, Any] = {}
    archetype = _pick_test_archetype()

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
            "category": archetype["category"],
            "country": archetype["country"],
            "name": archetype["name"],
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
    assert payload["building"]["name"] == archetype["name"]
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

    assert calls["building_name"] == archetype["name"]
    assert calls["weather_source"] == "pvgis"
    assert calls["sankey_graph"] is False
    assert calls["path_weather_file"] is None


def test_run_single_save_with_wall_window_pv_example_values(monkeypatch, tmp_path):
    """
    Run single-save with ECM wall+window and integrated PV.

    Uses an archetype loaded from building_examples.py and example values for:
    - u_wall
    - u_window
    - pv_kwp
    """

    archetype = _pick_test_archetype()
    calls: Dict[str, Any] = {}

    def _fake_iso_calc(
        bui,
        weather_source="pvgis",
        sankey_graph=False,
        path_weather_file=None,
    ):
        _ = bui
        calls["iso_weather_source"] = weather_source
        calls["iso_sankey_graph"] = sankey_graph
        calls["iso_path_weather_file"] = path_weather_file
        hourly = [
            {"Q_H_ideal": 1000.0, "Q_C_ideal": 100.0},
            {"Q_H_ideal": 900.0, "Q_C_ideal": 80.0},
        ]
        annual = [{"Q_H_annual": 1900.0, "Q_C_annual": 180.0}]
        return hourly, annual

    def _fake_integrated_pipeline(
        *,
        bui,
        pv,
        uni_cfg=None,
        return_hourly_building=False,
        hourly_sim=None,
        use_heat_pump=False,
        heat_pump_cop=3.2,
    ):
        calls["integrated"] = {
            "building_name": bui.get("building", {}).get("name"),
            "pv": pv,
            "uni_cfg_present": isinstance(uni_cfg, dict),
            "return_hourly_building": return_hourly_building,
            "hourly_len": len(hourly_sim) if hourly_sim is not None else 0,
            "use_heat_pump": use_heat_pump,
            "heat_pump_cop": heat_pump_cop,
        }
        return {
            "inputs": {
                "pv": {
                    "pv_kwp": float(pv["pv_kwp"]),
                    "tilt_deg": float(pv["tilt_deg"]),
                    "azimuth_deg": float(pv["azimuth_deg"]),
                }
            },
            "results": {
                "uni11300": {"summary": {"E_tot_annual_kWh": 1234.5}},
                "pv_hp": {"self_consumption_ratio": 0.42},
            },
        }

    monkeypatch.setattr(
        main_module.pybui.ISO52016,
        "Temperature_and_Energy_needs_calculation",
        _fake_iso_calc,
    )
    monkeypatch.setattr(
        main_module,
        "_run_iso52016_uni11300_pv_pipeline",
        _fake_integrated_pipeline,
    )

    output_dir = tmp_path / "ecm_single_out_wall_window_pv"
    bui_dir = tmp_path / "ecm_single_bui_wall_window_pv"
    u_wall = 0.35
    u_window = 1.1
    pv_kwp = 5.5

    response = client.post(
        "/ecm_application/run_single_save",
        params={
            "archetype": "true",
            "category": archetype["category"],
            "country": archetype["country"],
            "name": archetype["name"],
            "weather_source": "pvgis",
            "ecm_options": "wall,window",
            "u_wall": u_wall,
            "u_window": u_window,
            "include_baseline": "true",
            "use_pv": "true",
            "pv_kwp": pv_kwp,
            "pv_tilt_deg": 30.0,
            "pv_azimuth_deg": 0.0,
            "save_bui": "true",
            "output_dir": str(output_dir),
            "bui_dir": str(bui_dir),
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "completed"
    assert payload["building"]["name"] == archetype["name"]
    assert payload["use_pv"] is True
    assert payload["u_values_requested"]["wall"] == u_wall
    assert payload["u_values_requested"]["window"] == u_window
    assert set(payload["single_scenario_elements"]) == {"wall", "window"}
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["successful"] == 2
    assert payload["summary"]["failed"] == 0

    results_by_id = {item["scenario_id"]: item for item in payload["results"]}
    assert {"baseline", "single"} <= set(results_by_id.keys())

    baseline = results_by_id["baseline"]
    assert baseline["status"] == "success"
    assert baseline["combo"] == []
    assert baseline["integrated_results"] is None
    assert Path(baseline["files"]["hourly_csv"]).exists()
    assert Path(baseline["files"]["annual_csv"]).exists()
    assert Path(baseline["files"]["bui_json"]).exists()

    single = results_by_id["single"]
    assert single["status"] == "success"
    assert set(single["combo"]) == {"pv", "wall", "window"}
    assert single["integrated_results"]["inputs"]["pv"]["pv_kwp"] == pv_kwp
    assert Path(single["files"]["hourly_csv"]).exists()
    assert Path(single["files"]["annual_csv"]).exists()
    assert Path(single["files"]["bui_json"]).exists()

    assert calls["iso_weather_source"] == "pvgis"
    assert calls["iso_sankey_graph"] is False
    assert calls["iso_path_weather_file"] is None
    assert calls["integrated"]["building_name"] == archetype["name"]
    assert calls["integrated"]["pv"]["pv_kwp"] == pv_kwp
    assert calls["integrated"]["pv"]["tilt_deg"] == 30.0
    assert calls["integrated"]["pv"]["azimuth_deg"] == 0.0
    assert calls["integrated"]["return_hourly_building"] is False
    assert calls["integrated"]["use_heat_pump"] is False


if __name__ == "__main__":
    from run_single_save_main import main as run_single_save_main

    raise SystemExit(run_single_save_main())
