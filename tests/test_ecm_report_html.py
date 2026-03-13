import json

import pandas as pd
from fastapi.testclient import TestClient

import relife_forecasting.main as main_module
from relife_forecasting.utils.ecm_report_html import (
    _compute_uni_from_building_hourly,
    build_ecm_comparison_report_html,
)


client = TestClient(main_module.app)


def _build_custom_bui() -> dict:
    return {
        "building": {
            "name": "Report BUI",
            "building_type_class": "Residential_single_family",
        },
        "building_surface": [
            {
                "name": "Opaque south surface",
                "type": "opaque",
                "area": 12.0,
                "sky_view_factor": 0.5,
                "u_value": 1.8,
                "orientation": {"tilt": 90, "azimuth": 180},
            },
            {
                "name": "Transparent south surface",
                "type": "transparent",
                "area": 4.0,
                "sky_view_factor": 0.5,
                "u_value": 2.7,
                "g_value": 0.6,
                "height": 1.4,
                "width": 1.4,
                "orientation": {"tilt": 90, "azimuth": 180},
            },
        ],
        "building_parameters": {},
    }


def _fake_iso_with_wall_effect(calls):
    def _runner(
        bui,
        weather_source="pvgis",
        path_weather_file=None,
        sankey_graph=False,
    ):
        wall_u = next(
            surface["u_value"]
            for surface in bui["building_surface"]
            if main_module.classify_surface(surface) == "wall"
        )
        calls.append(
            {
                "weather_source": weather_source,
                "path_weather_file": path_weather_file,
                "sankey_graph": sankey_graph,
                "wall_u": wall_u,
            }
        )

        idx = pd.date_range("2025-01-01 00:00:00", periods=48, freq="h")
        heat_wh = 1000.0 if wall_u > 1.0 else 500.0
        op_temp = 20.0 if wall_u > 1.0 else 21.0
        hourly = pd.DataFrame(
            {
                "T_ext": [6.0] * len(idx),
                "T_op": [op_temp] * len(idx),
                "Q_H": [heat_wh] * len(idx),
                "Q_C": [0.0] * len(idx),
            },
            index=idx,
        )
        annual = pd.DataFrame(
            [
                {
                    "Q_H_annual": float(hourly["Q_H"].sum() * 0.001),
                    "Q_C_annual": float(hourly["Q_C"].sum() * 0.001),
                }
            ]
        )
        return hourly, annual

    return _runner


def _fake_uni_results(
    *,
    hourly_sim,
    uni_cfg,
    generation_mask,
    heat_pump_cop,
):
    _ = uni_cfg, heat_pump_cop
    hourly_df = hourly_sim.copy() if isinstance(hourly_sim, pd.DataFrame) else pd.DataFrame(hourly_sim)
    q_heat_kwh = pd.to_numeric(hourly_df["Q_H"], errors="coerce").fillna(0.0) * 0.001

    multiplier = 1.0
    if generation_mask.get("applied_mode") == "heat_pump":
        multiplier = 0.5
    elif generation_mask.get("requested_mode") == "condensing_boiler":
        multiplier = 0.8

    ep_heat = q_heat_kwh * multiplier
    hourly_uni = pd.DataFrame(
        {
            "EP_heat_total_kWh": ep_heat,
            "EP_cool_total_kWh": [0.0] * len(hourly_df),
            "EP_total_kWh": ep_heat,
        },
        index=hourly_df.index,
    )

    return {
        "hourly_results": main_module.dataframe_to_records_safe(hourly_uni),
        "summary": {
            "EP_heat_total_kWh": float(ep_heat.sum()),
            "EP_cool_total_kWh": 0.0,
            "EP_total_kWh": float(ep_heat.sum()),
        },
        "generation_mask": generation_mask,
    }


def test_ecm_application_report_renders_html_with_baseline_vs_ecm(monkeypatch):
    calls = []

    monkeypatch.setattr(main_module, "run_iso52016_simulation", _fake_iso_with_wall_effect(calls))
    monkeypatch.setattr(main_module, "_compute_ecm_uni11300_results", _fake_uni_results)
    monkeypatch.setattr(main_module, "json_to_internal_system", lambda payload: payload)

    response = client.post(
        "/ecm_application/report",
        params={
            "archetype": "false",
            "weather_source": "pvgis",
            "u_wall": 0.25,
            "scenario_elements": "wall",
            "use_heat_pump": "true",
            "heat_pump_cop": 4.0,
            "report_title": "Test ECM HTML",
        },
        data={
            "bui_json": json.dumps(_build_custom_bui()),
            "system_json": json.dumps({"generator_type": "boiler"}),
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/html")
    assert len(calls) == 2
    assert calls[0]["wall_u"] == 1.8
    assert calls[1]["wall_u"] == 0.25

    html = response.text
    assert "<title>Test ECM HTML</title>" in html
    assert "echarts.min.js" in html
    assert "Hourly temperature trends" in html
    assert "Monthly ISO 52016 energy needs" in html
    assert "Monthly UNI/TS 11300 primary energy" in html
    assert "Annual comparison and savings" in html
    assert "Scenario summary" in html
    assert "ECM options" in html
    assert "Generation setup" in html
    assert "Eta / COP" in html
    assert "ISO 52016 saving [%]" in html
    assert "Primary energy saving [%]" in html
    assert "walls U=0.25 + heat pump COP=4.0" in html
    assert "heat_pump" in html
    assert "75.0%" in html
    assert "2025-01" in html
    assert "Indivisible" in html
    assert "grid-template-columns: 1fr;" in html


def test_ecm_application_report_generation_only_reuses_baseline_iso(monkeypatch):
    calls = []

    monkeypatch.setattr(main_module, "run_iso52016_simulation", _fake_iso_with_wall_effect(calls))
    monkeypatch.setattr(main_module, "_compute_ecm_uni11300_results", _fake_uni_results)

    response = client.post(
        "/ecm_application/report",
        params={
            "archetype": "false",
            "weather_source": "pvgis",
            "uni_generation_mode": "condensing_boiler",
            "report_title": "Generation Only Report",
        },
        data={"bui_json": json.dumps(_build_custom_bui())},
    )

    assert response.status_code == 200, response.text
    assert len(calls) == 1

    html = response.text
    assert "<title>Generation Only Report</title>" in html
    assert "Scenario summary" in html
    assert "generation only" in html
    assert "condensing boiler eta_generation=1.1" in html
    assert "20.0%" in html


def test_report_recomputes_uni_monthly_from_hourly_building_when_payload_is_stale():
    idx = pd.date_range("2025-01-01 00:00:00", periods=24 * 31, freq="h")
    baseline_hourly = pd.DataFrame(
        {
            "T_ext": [6.0] * len(idx),
            "T_op": [20.0] * len(idx),
            "Q_H": [1000.0] * len(idx),
            "Q_C": [0.0] * len(idx),
        },
        index=idx,
    )
    scenario_hourly = baseline_hourly.copy()
    scenario_hourly["Q_H"] = 500.0
    scenario_hourly["T_op"] = 21.0

    baseline_mask = {"requested_mode": "default", "applied_mode": "default"}
    scenario_mask = {"requested_mode": "default", "applied_mode": "heat_pump", "mask_value": 4.0, "heat_pump_cop": 4.0}

    baseline_expected_hourly, baseline_expected_summary = _compute_uni_from_building_hourly(
        baseline_hourly,
        generation_mask=baseline_mask,
    )
    scenario_expected_hourly, scenario_expected_summary = _compute_uni_from_building_hourly(
        scenario_hourly,
        generation_mask=scenario_mask,
    )
    baseline_expected_monthly = baseline_expected_hourly["EP_total_kWh"].resample("ME").sum().iloc[0]
    scenario_expected_monthly = scenario_expected_hourly["EP_total_kWh"].resample("ME").sum().iloc[0]

    html = build_ecm_comparison_report_html(
        report_title="Recomputed UNI Report",
        building_meta={"name": "B1", "category": "Residential", "country": "GRC"},
        scenario_meta={
            "id": "hp",
            "label": "heat pump COP=4.0",
            "elements": [],
            "generation_mask": scenario_mask,
        },
        baseline_hourly=baseline_hourly,
        scenario_hourly=scenario_hourly,
        baseline_uni_results={
            "hourly_results": [
                {"timestamp": "2025-01-31 23:00:00", "EP_heat_total_kWh": 99999.0, "EP_cool_total_kWh": 0.0, "EP_total_kWh": 99999.0}
            ],
            "summary": {"EP_heat_total_kWh": 99999.0, "EP_cool_total_kWh": 0.0, "EP_total_kWh": 99999.0},
            "generation_mask": baseline_mask,
        },
        scenario_uni_results={
            "hourly_results": [
                {"timestamp": "2025-01-31 23:00:00", "EP_heat_total_kWh": 50000.0, "EP_cool_total_kWh": 0.0, "EP_total_kWh": 50000.0}
            ],
            "summary": {"EP_heat_total_kWh": 50000.0, "EP_cool_total_kWh": 0.0, "EP_total_kWh": 50000.0},
            "generation_mask": scenario_mask,
        },
    )

    assert "99,999.00" not in html
    assert "50,000.00" not in html
    assert f"{baseline_expected_monthly:,.2f}" in html
    assert f"{scenario_expected_monthly:,.2f}" in html
    assert f"{baseline_expected_summary['EP_total_kWh']:,.2f}" in html
    assert f"{scenario_expected_summary['EP_total_kWh']:,.2f}" in html
