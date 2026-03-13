import json

from fastapi.testclient import TestClient

import relife_forecasting.main as main_module


client = TestClient(main_module.app)


def _pick_test_archetype_with_uni() -> dict:
    archetype = next(
        (
            arch
            for arch in main_module.BUILDING_ARCHETYPES
            if isinstance(arch.get("category"), str)
            and isinstance(arch.get("country"), str)
            and isinstance(arch.get("name"), str)
            and isinstance(arch.get("uni11300"), dict)
            and isinstance(arch.get("system"), dict)
        ),
        None,
    )
    if archetype is None:
        raise AssertionError("No archetype with uni11300/system configuration available for ECM tests.")
    return archetype


def test_ecm_application_custom_bui_runs_only_requested_single_scenario(monkeypatch):
    """Exercise /ecm_application in single-scenario mode for the requested ECM set."""

    calls = []

    def _fake_iso_calc(
        bui,
        weather_source="pvgis",
        path_weather_file=None,
        sankey_graph=False,
    ):
        calls.append(
            {
                "weather_source": weather_source,
                "path_weather_file": path_weather_file,
                "sankey_graph": sankey_graph,
                "wall_u": next(
                    surface["u_value"]
                    for surface in bui["building_surface"]
                    if surface["name"] == "Opaque south surface"
                ),
                "window_u": next(
                    surface["u_value"]
                    for surface in bui["building_surface"]
                    if surface["name"] == "Transparent south surface"
                ),
                "roof_u": next(
                    surface["u_value"]
                    for surface in bui["building_surface"]
                    if surface["name"] == "Roof surface"
                ),
            }
        )
        return [{"Q_H": 1000.0, "Q_C": 100.0}], [{"Q_H_annual": 1000.0}]

    monkeypatch.setattr(main_module, "run_iso52016_simulation", _fake_iso_calc)

    bui_payload = {
        "building": {
            "name": "Custom BUI",
            "building_type_class": "Residential_single_family",
        },
        "building_surface": [
            {
                "name": "Roof surface",
                "type": "opaque",
                "area": 20.0,
                "sky_view_factor": 1.0,
                "u_value": 1.5,
                "orientation": {"tilt": 0, "azimuth": 0},
            },
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

    response = client.post(
        "/ecm_application",
        params={
            "archetype": "false",
            "weather_source": "pvgis",
            "u_wall": 0.25,
            "u_window": 1.1,
            "scenario_elements": "wall,window",
        },
        data={"bui_json": json.dumps(bui_payload)},
    )

    assert response.status_code == 200, response.text

    payload = response.json()

    assert payload["source"] == "custom"
    assert payload["weather_source"] == "pvgis"
    assert payload["u_values_requested"]["wall"] == 0.25
    assert payload["u_values_requested"]["window"] == 1.1
    assert payload["single_scenario_mode"]["scenario_elements"] == "wall,window"
    assert payload["single_scenario_mode"]["baseline_only"] is False
    assert payload["single_scenario_mode"]["scenario_id"] is None
    assert payload["n_scenarios"] == 1
    assert len(payload["scenarios"]) == 1

    scenario = payload["scenarios"][0]
    assert scenario["scenario_id"] == "wall+window"
    assert scenario["elements"] == ["wall", "window"]
    assert scenario["u_values"] == {
        "roof": None,
        "wall": 0.25,
        "window": 1.1,
        "slab": None,
    }
    assert "results" in scenario
    assert "annual_building" in scenario["results"]
    assert "hourly_building" in scenario["results"]
    assert "primary_energy_uni11300" in scenario["results"]
    assert "summary" in scenario["results"]["primary_energy_uni11300"]
    assert payload["uni11300_generation_mask"]["applied_mode"] == "default"

    assert len(calls) == 1
    assert calls[0]["weather_source"] == "pvgis"
    assert calls[0]["path_weather_file"] is None
    assert calls[0]["sankey_graph"] is False
    assert calls[0]["wall_u"] == 0.25
    assert calls[0]["window_u"] == 1.1
    assert calls[0]["roof_u"] == 1.5


def test_ecm_application_applies_condensing_boiler_generation_mask(monkeypatch):
    """Set eta_generation=1.1 in UNI/TS 11300 when condensing boiler mode is requested."""

    captured = {}

    def _fake_iso_calc(
        bui,
        weather_source="pvgis",
        path_weather_file=None,
        sankey_graph=False,
    ):
        _ = bui, weather_source, path_weather_file, sankey_graph
        return [{"Q_H": 1000.0, "Q_C": 0.0}], [{"Q_H_annual": 1000.0}]

    def _fake_uni11300(
        hourly_df,
        input_unit="Wh",
        heating_params_payload=None,
        cooling_params_payload=None,
    ):
        captured["input_unit"] = input_unit
        captured["heating_params_payload"] = dict(heating_params_payload or {})
        captured["cooling_params_payload"] = dict(cooling_params_payload or {})
        assert not hourly_df.empty
        return {
            "input_unit": input_unit,
            "ideal_unit": "kWh",
            "n_hours": len(hourly_df),
            "hourly_results": [{"EP_total_kWh": 1.0}],
            "summary": {"EP_total_kWh": 1.0},
        }

    monkeypatch.setattr(main_module, "run_iso52016_simulation", _fake_iso_calc)
    monkeypatch.setattr(main_module, "_compute_uni11300_full_from_hourly_df", _fake_uni11300)

    bui_payload = {
        "building": {
            "name": "Custom BUI",
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
            }
        ],
        "building_parameters": {},
    }

    response = client.post(
        "/ecm_application",
        params={
            "archetype": "false",
            "weather_source": "pvgis",
            "u_wall": 0.25,
            "scenario_elements": "wall",
            "uni_generation_mode": "condensing_boiler",
        },
        data={"bui_json": json.dumps(bui_payload)},
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    assert captured["input_unit"].lower() == "wh"
    assert captured["heating_params_payload"]["eta_generation"] == 1.1
    assert payload["uni11300_generation_mask"]["requested_mode"] == "condensing_boiler"
    assert payload["uni11300_generation_mask"]["applied_mode"] == "condensing_boiler"
    assert payload["uni11300_generation_mask"]["metric"] == "eta_generation"
    assert payload["uni11300_generation_mask"]["mask_value"] == 1.1
    assert (
        payload["scenarios"][0]["results"]["primary_energy_uni11300"]["generation_mask"]["mask_value"]
        == 1.1
    )


def test_ecm_application_applies_heat_pump_cop_to_uni11300(monkeypatch):
    """When heat pump ECM is enabled, UNI/TS 11300 uses the ECM COP mask."""

    archetype = _pick_test_archetype_with_uni()
    captured = {}

    def _fake_iso_calc(
        bui,
        weather_source="pvgis",
        path_weather_file=None,
        sankey_graph=False,
    ):
        _ = bui, weather_source, path_weather_file, sankey_graph
        return [{"Q_H": 1000.0, "Q_C": 150.0}], [{"Q_H_annual": 1000.0}]

    def _fake_uni11300(
        hourly_df,
        input_unit="Wh",
        heating_params_payload=None,
        cooling_params_payload=None,
    ):
        captured["heating_params_payload"] = dict(heating_params_payload or {})
        return {
            "input_unit": input_unit,
            "ideal_unit": "kWh",
            "n_hours": len(hourly_df),
            "hourly_results": [
                {
                    "Q_ideal_heat_kWh": 1.0,
                    "Q_ideal_cool_kWh": 0.15,
                    "E_aux_total_heat_kWh": 0.05,
                    "E_delivered_electric_heat_kWh": 0.05,
                    "E_delivered_electric_cool_kWh": 0.10,
                    "EP_cool_total_kWh": 0.25,
                }
            ],
            "summary": {"EP_total_kWh": 1.0},
        }

    def _fake_apply_heat_pump(uni11300_results, heat_pump_cop, fp_electric=2.18):
        captured["heat_pump_cop"] = heat_pump_cop
        captured["fp_electric"] = fp_electric
        out = dict(uni11300_results)
        out["heat_pump_applied"] = True
        out["heat_pump_cop"] = heat_pump_cop
        return out

    monkeypatch.setattr(main_module, "run_iso52016_simulation", _fake_iso_calc)
    monkeypatch.setattr(main_module, "_compute_uni11300_full_from_hourly_df", _fake_uni11300)
    monkeypatch.setattr(main_module, "_apply_heat_pump_to_uni11300_results", _fake_apply_heat_pump)

    response = client.post(
        "/ecm_application",
        params={
            "archetype": "true",
            "category": archetype["category"],
            "country": archetype["country"],
            "name": archetype["name"],
            "weather_source": "pvgis",
            "u_wall": 0.25,
            "scenario_elements": "wall",
            "use_heat_pump": "true",
            "heat_pump_cop": 4.1,
            "uni_generation_mode": "condensing_boiler",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    assert captured["heat_pump_cop"] == 4.1
    assert captured["fp_electric"] == archetype["uni11300"]["heating_params"]["fp_electric"]
    assert payload["uni11300_generation_mask"]["requested_mode"] == "condensing_boiler"
    assert payload["uni11300_generation_mask"]["applied_mode"] == "heat_pump"
    assert payload["uni11300_generation_mask"]["metric"] == "cop_generation"
    assert payload["uni11300_generation_mask"]["mask_value"] == 4.1
    assert payload["scenarios"][0]["results"]["primary_energy_uni11300"]["heat_pump_applied"] is True
