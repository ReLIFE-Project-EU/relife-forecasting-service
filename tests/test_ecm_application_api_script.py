from pathlib import Path

import pandas as pd

from relife_forecasting.analysis.ECM import ecm_application_api as api_module


def _hourly_payload(q_heat_wh: float, ep_total_kwh: float, t_op: float) -> dict:
    hourly_rows = [
        {
            "timestamp": "2025-01-01 00:00:00",
            "T_ext": 6.0,
            "T_op": t_op,
            "Q_H": q_heat_wh,
            "Q_C": 0.0,
        },
        {
            "timestamp": "2025-01-01 01:00:00",
            "T_ext": 5.5,
            "T_op": t_op,
            "Q_H": q_heat_wh,
            "Q_C": 0.0,
        },
    ]
    uni_rows = [
        {
            "timestamp": "2025-01-01 00:00:00",
            "EP_heat_total_kWh": ep_total_kwh / 2.0,
            "EP_cool_total_kWh": 0.0,
            "EP_total_kWh": ep_total_kwh / 2.0,
        },
        {
            "timestamp": "2025-01-01 01:00:00",
            "EP_heat_total_kWh": ep_total_kwh / 2.0,
            "EP_cool_total_kWh": 0.0,
            "EP_total_kWh": ep_total_kwh / 2.0,
        },
    ]
    return {
        "hourly_building": hourly_rows,
        "primary_energy_uni11300": {
            "hourly_results": uni_rows,
            "summary": {
                "EP_heat_total_kWh": ep_total_kwh,
                "EP_cool_total_kWh": 0.0,
                "EP_total_kWh": ep_total_kwh,
            },
            "generation_mask": {
                "requested_mode": "default",
                "applied_mode": "default",
                "metric": "eta_generation",
                "mask_value": None,
            },
        },
    }


def test_main_builds_report_locally_without_recalling_report_endpoint(monkeypatch, tmp_path):
    baseline_context = {
        "building_meta": {
            "name": "Script Report BUI",
            "category": "Residential_single_family",
            "country": "Greece",
            "weather_source": "pvgis",
        },
        "scenario_meta": {
            "id": "baseline",
            "label": "Baseline (no changes)",
            "elements": [],
            "generation_mask": {
                "requested_mode": "default",
                "applied_mode": "default",
                "metric": "eta_generation",
                "mask_value": None,
            },
        },
        **_hourly_payload(q_heat_wh=1000.0, ep_total_kwh=2.0, t_op=20.0),
    }
    scenario_context = {
        "building_meta": baseline_context["building_meta"],
        "scenario_meta": {
            "id": "wall",
            "label": "walls U=0.25",
            "description": "Wall insulation upgrade",
            "combo": ["wall"],
            "elements": ["wall"],
            "generation_mask": {
                "requested_mode": "default",
                "applied_mode": "default",
                "metric": "eta_generation",
                "mask_value": None,
            },
        },
        **_hourly_payload(q_heat_wh=500.0, ep_total_kwh=1.0, t_op=21.0),
    }
    fake_stats = {
        "total": 2,
        "successful": 2,
        "failed": 0,
        "total_time": 0.2,
        "results": [
            {
                "status": "success",
                "combo": [],
                "combo_tag": "baseline",
                "scenario_id": "baseline",
                "elapsed": 0.1,
                "size_kb": 1.0,
                "file_hourly": str(tmp_path / "baseline.csv"),
                "file_annual": str(tmp_path / "baseline__annual.csv"),
                "report_context": baseline_context,
            },
            {
                "status": "success",
                "combo": ["wall"],
                "combo_tag": "wall",
                "scenario_id": "wall",
                "elapsed": 0.1,
                "size_kb": 1.0,
                "file_hourly": str(tmp_path / "wall.csv"),
                "file_annual": str(tmp_path / "wall__annual.csv"),
                "report_context": scenario_context,
            },
        ],
        "scenario_combo": ["wall"],
    }

    monkeypatch.setattr(api_module, "run_ecm_api_sequential", lambda **_: fake_stats)
    monkeypatch.setattr(api_module, "build_summary_table", lambda results: pd.DataFrame(results))
    monkeypatch.setattr(
        api_module,
        "call_ecm_report",
        lambda **_: (_ for _ in ()).throw(AssertionError("call_ecm_report must not be used")),
    )
    monkeypatch.setattr(api_module, "USE_RENOVATION_SCENARIO_LIBRARY", False)
    monkeypatch.setattr(api_module, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(api_module, "WEATHER_SOURCE", "pvgis")
    monkeypatch.setattr(api_module, "EPW_PATH", None)
    monkeypatch.setattr(api_module, "ECM_OPTIONS", ["wall"])
    monkeypatch.setattr(api_module, "USE_HEAT_PUMP", False)
    monkeypatch.setattr(api_module, "REPORT_TITLE", "Local ECM Report")
    monkeypatch.setattr(api_module, "GENERATE_REPORT", True)
    monkeypatch.setattr(api_module, "ARCHETYPE_NAME", "Script Report BUI")

    outcome = api_module.main()

    assert outcome["report_error"] is None
    assert outcome["report_path"] is not None

    report_path = Path(outcome["report_path"])
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "<title>Local ECM Report</title>" in html
    assert "walls U=0.25" in html
    assert "Scenario summary" in html
    assert "Wall insulation upgrade" in html
    assert "ISO 52016 saving [%]" in html
    assert "Primary energy saving [%]" in html
    assert "Indivisible" in html
    assert "grid-template-columns: 1fr;" in html


def test_main_library_builds_multi_scenario_report(monkeypatch, tmp_path):
    baseline_context = {
        "building_meta": {
            "name": "Scenario Library BUI",
            "category": "Residential_single_family",
            "country": "Greece",
            "weather_source": "pvgis",
        },
        "scenario_meta": {
            "id": "baseline",
            "label": "Baseline (no changes)",
            "elements": [],
            "generation_mask": {
                "requested_mode": "default",
                "applied_mode": "default",
                "metric": "eta_generation",
                "mask_value": None,
            },
        },
        **_hourly_payload(q_heat_wh=1000.0, ep_total_kwh=2.0, t_op=20.0),
    }
    deep_envelope_context = {
        "building_meta": baseline_context["building_meta"],
        "scenario_meta": {
            "id": "deep_envelope",
            "label": "deep_envelope",
            "description": "Deep envelope retrofit",
            "combo": ["wall", "window", "roof", "slab"],
            "elements": ["wall", "window", "roof", "slab"],
            "generation_mask": {
                "requested_mode": "default",
                "applied_mode": "default",
                "metric": "eta_generation",
                "mask_value": None,
            },
        },
        **_hourly_payload(q_heat_wh=600.0, ep_total_kwh=1.2, t_op=20.8),
    }
    condensing_boiler_context = {
        "building_meta": baseline_context["building_meta"],
        "scenario_meta": {
            "id": "condensing_boiler",
            "label": "condensing_boiler",
            "description": "Condensing boiler replacement",
            "combo": ["condensing_boiler"],
            "elements": [],
            "generation_mask": {
                "requested_mode": "condensing_boiler",
                "applied_mode": "condensing_boiler",
                "metric": "eta_generation",
                "mask_value": 1.1,
            },
        },
        **_hourly_payload(q_heat_wh=800.0, ep_total_kwh=1.5, t_op=20.2),
    }
    condensing_boiler_context["primary_energy_uni11300"]["generation_mask"] = {
        "requested_mode": "condensing_boiler",
        "applied_mode": "condensing_boiler",
        "metric": "eta_generation",
        "mask_value": 1.1,
    }

    fake_stats = {
        "total": 3,
        "successful": 3,
        "failed": 0,
        "total_time": 0.3,
        "results": [
            {
                "status": "success",
                "scenario_name": "baseline",
                "combo": [],
                "combo_tag": "baseline",
                "scenario_id": "baseline",
                "elapsed": 0.1,
                "size_kb": 1.0,
                "file_hourly": str(tmp_path / "baseline.csv"),
                "file_annual": str(tmp_path / "baseline__annual.csv"),
                "report_context": baseline_context,
            },
            {
                "status": "success",
                "scenario_name": "deep_envelope",
                "combo": ["wall", "window", "roof", "slab"],
                "combo_tag": "roof_slab_wall_window",
                "scenario_id": "deep_envelope",
                "elapsed": 0.1,
                "size_kb": 1.0,
                "file_hourly": str(tmp_path / "deep_envelope.csv"),
                "file_annual": str(tmp_path / "deep_envelope__annual.csv"),
                "report_context": deep_envelope_context,
            },
            {
                "status": "success",
                "scenario_name": "condensing_boiler",
                "combo": ["condensing_boiler"],
                "combo_tag": "condensing_boiler",
                "scenario_id": "condensing_boiler",
                "elapsed": 0.1,
                "size_kb": 1.0,
                "file_hourly": str(tmp_path / "condensing_boiler.csv"),
                "file_annual": str(tmp_path / "condensing_boiler__annual.csv"),
                "report_context": condensing_boiler_context,
            },
        ],
        "scenario_combo": [],
        "scenario_names": ["deep_envelope", "condensing_boiler"],
    }

    monkeypatch.setattr(api_module, "run_predefined_renovation_scenarios", lambda **_: fake_stats)
    monkeypatch.setattr(api_module, "build_summary_table", lambda results: pd.DataFrame(results))
    monkeypatch.setattr(api_module, "USE_RENOVATION_SCENARIO_LIBRARY", True)
    monkeypatch.setattr(api_module, "REPORT_SCENARIO_NAME", None)
    monkeypatch.setattr(api_module, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(api_module, "WEATHER_SOURCE", "pvgis")
    monkeypatch.setattr(api_module, "EPW_PATH", None)
    monkeypatch.setattr(api_module, "REPORT_TITLE", "Scenario Library Report")
    monkeypatch.setattr(api_module, "GENERATE_REPORT", True)
    monkeypatch.setattr(api_module, "ARCHETYPE_NAME", "Scenario Library BUI")

    outcome = api_module.main()

    assert outcome["report_error"] is None
    assert outcome["report_path"] is not None

    report_path = Path(outcome["report_path"])
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "<title>Scenario Library Report</title>" in html
    assert "Scenario summary" in html
    assert "Deep envelope retrofit" in html
    assert "Condensing boiler replacement" in html
    assert "ECM options" in html
    assert "Generation setup" in html
    assert "Eta / COP" in html
    assert "deep_envelope Q_H" in html
    assert "condensing_boiler + condensing boiler eta_generation=1.1" in html
    assert "condensing_boiler + condensing boiler eta_generation=1.1 Q_H" in html
    assert "40.0%" in html
    assert "20.0%" in html


def test_renovation_scenario_library_includes_pv_and_condensing_boiler():
    scenarios = {scenario.name: scenario for scenario in api_module.RENOVATION_SCENARIOS}

    assert "condensing_boiler" in scenarios
    assert scenarios["condensing_boiler"].uni_generation_mode == "condensing_boiler"

    assert "pv_only" in scenarios
    assert scenarios["pv_only"].use_pv is True

    assert "deep_retrofit_hp_pv" in scenarios
    assert scenarios["deep_retrofit_hp_pv"].use_heat_pump is True
    assert scenarios["deep_retrofit_hp_pv"].use_pv is True


def test_run_predefined_renovation_scenarios_routes_pv_and_ecm(monkeypatch, tmp_path):
    calls = []

    def _fake_ecm(task):
        calls.append(("ecm", task.scenario_name, list(task.combo)))
        return {
            "status": "success",
            "scenario_name": task.scenario_name,
            "combo": task.combo,
            "combo_tag": "_".join(task.combo) if task.combo else "baseline",
            "scenario_id": task.scenario_name,
            "elapsed": 0.1,
            "size_kb": 1.0,
            "runner_type": "ecm_application",
        }

    def _fake_pv(task):
        calls.append(("pv", task.scenario_name, list(task.combo)))
        return {
            "status": "success",
            "scenario_name": task.scenario_name,
            "combo": task.combo,
            "combo_tag": "_".join(task.combo) if task.combo else "baseline",
            "scenario_id": task.scenario_name,
            "elapsed": 0.1,
            "size_kb": 1.0,
            "runner_type": "iso52016_uni11300_pv",
        }

    monkeypatch.setattr(api_module, "call_ecm_application", _fake_ecm)
    monkeypatch.setattr(api_module, "call_iso52016_uni11300_pv", _fake_pv)

    selected_scenarios = [
        api_module.RenovationScenario(
            name="condensing_boiler",
            description="Condensing boiler replacement",
            ecm_options=[],
            uni_generation_mode="condensing_boiler",
        ),
        api_module.RenovationScenario(
            name="pv_only",
            description="Photovoltaic system installation",
            ecm_options=[],
            use_pv=True,
            pv_config=api_module.DEFAULT_PV_CONFIG,
        ),
    ]

    stats = api_module.run_predefined_renovation_scenarios(
        scenarios=selected_scenarios,
        weather_source="pvgis",
        epw_path=None,
        output_dir=tmp_path,
        include_baseline=True,
        archetype=True,
        category="Single Family House",
        country="Greece",
        name="SFH_Greece_1946_1969",
    )

    assert stats["successful"] == 3
    assert calls == [
        ("ecm", "baseline", []),
        ("ecm", "condensing_boiler", ["condensing_boiler"]),
        ("pv", "pv_only", ["pv"]),
    ]
