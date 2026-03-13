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


def _find_archetype(archetype: Dict[str, str]) -> Dict[str, Any]:
    match = next(
        (
            arch
            for arch in main_module.BUILDING_ARCHETYPES
            if arch.get("category") == archetype["category"]
            and arch.get("country") == archetype["country"]
            and arch.get("name") == archetype["name"]
        ),
        None,
    )
    if match is None:
        raise AssertionError(f"Archetype not found: {archetype}")
    return match


def _representative_u_values(bui: Dict[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for surface in bui.get("building_surface", []):
        kind = main_module.classify_surface(surface)
        if kind and kind not in values:
            values[kind] = float(surface["u_value"])
    return values


def test_run_sequential_save_with_baseline_and_combinations(monkeypatch, tmp_path):
    """Run the sequential-save endpoint and verify the generated baseline + ECM combinations."""

    archetype = _pick_test_archetype()
    match = _find_archetype(archetype)
    original_u = _representative_u_values(match["bui"])
    calls = []

    def _fake_iso_calc(
        bui,
        weather_source="pvgis",
        sankey_graph=False,
        path_weather_file=None,
    ):
        calls.append(
            {
                "building_name": bui.get("building", {}).get("name"),
                "weather_source": weather_source,
                "sankey_graph": sankey_graph,
                "path_weather_file": path_weather_file,
                "u_values": _representative_u_values(bui),
            }
        )
        hourly = [
            {"Q_H": 1200.0, "Q_C": 150.0},
            {"Q_H": 1100.0, "Q_C": 120.0},
        ]
        annual = [{"Q_H_annual": 2300.0, "Q_C_annual": 270.0}]
        return hourly, annual

    monkeypatch.setattr(main_module, "run_iso52016_simulation", _fake_iso_calc)

    output_dir = tmp_path / "ecm_seq_out"
    bui_dir = tmp_path / "ecm_seq_bui"
    u_wall = 0.5
    u_window = 1.0

    response = client.post(
        "/ecm_application/run_sequential_save",
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
            "save_bui": "true",
            "output_dir": str(output_dir),
            "bui_dir": str(bui_dir),
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "completed"
    assert payload["source"] == "archetype"
    assert payload["building"] == archetype
    assert payload["weather_source"] == "pvgis"
    assert payload["u_values_requested"] == {
        "roof": None,
        "wall": u_wall,
        "window": u_window,
        "slab": None,
    }
    assert payload["ecm_options"] == ["wall", "window"]
    assert payload["include_baseline"] is True
    assert payload["output_dir"] == str(output_dir)
    assert payload["bui_dir"] == str(bui_dir)

    summary = payload["summary"]
    assert summary["total"] == 4
    assert summary["successful"] == 4
    assert summary["failed"] == 0
    assert summary["total_time_s"] >= 0

    results = payload["results"]
    assert [item["combo"] for item in results] == [
        [],
        ["wall"],
        ["window"],
        ["wall", "window"],
    ]
    assert [item["combo_tag"] for item in results] == [
        "BASELINE",
        "wall",
        "window",
        "wall,window",
    ]

    for item in results:
        assert item["status"] == "success"
        assert item["elapsed_s"] >= 0
        files = item["files"]
        assert Path(files["hourly_csv"]).exists()
        assert Path(files["annual_csv"]).exists()
        assert Path(files["bui_json"]).exists()

    assert len(calls) == 4
    assert all(call["building_name"] == archetype["name"] for call in calls)
    assert all(call["weather_source"] == "pvgis" for call in calls)
    assert all(call["sankey_graph"] is False for call in calls)
    assert all(call["path_weather_file"] is None for call in calls)

    assert calls[0]["u_values"]["wall"] == original_u["wall"]
    assert calls[0]["u_values"]["window"] == original_u["window"]

    assert calls[1]["u_values"]["wall"] == u_wall
    assert calls[1]["u_values"]["window"] == original_u["window"]

    assert calls[2]["u_values"]["wall"] == original_u["wall"]
    assert calls[2]["u_values"]["window"] == u_window

    assert calls[3]["u_values"]["wall"] == u_wall
    assert calls[3]["u_values"]["window"] == u_window


def test_run_sequential_save_infers_ecm_options_from_u_values(monkeypatch, tmp_path):
    """If ecm_options is omitted, the endpoint should infer the active combinations from provided U-values."""

    archetype = _pick_test_archetype()
    calls = []

    def _fake_iso_calc(
        bui,
        weather_source="pvgis",
        sankey_graph=False,
        path_weather_file=None,
    ):
        _ = bui, weather_source, sankey_graph, path_weather_file
        calls.append("iso")
        return [{"Q_H": 900.0, "Q_C": 50.0}], [{"Q_H_annual": 900.0}]

    monkeypatch.setattr(main_module, "run_iso52016_simulation", _fake_iso_calc)

    response = client.post(
        "/ecm_application/run_sequential_save",
        params={
            "archetype": "true",
            "category": archetype["category"],
            "country": archetype["country"],
            "name": archetype["name"],
            "weather_source": "pvgis",
            "u_wall": 0.4,
            "include_baseline": "false",
            "save_bui": "false",
            "output_dir": str(tmp_path / "ecm_seq_inferred_out"),
            "bui_dir": str(tmp_path / "ecm_seq_inferred_bui"),
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["ecm_options"] == ["wall"]
    assert payload["include_baseline"] is False
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["successful"] == 1
    assert payload["summary"]["failed"] == 0
    assert len(payload["results"]) == 1
    assert payload["results"][0]["combo"] == ["wall"]
    assert payload["results"][0]["combo_tag"] == "wall"
    assert payload["results"][0]["files"]["bui_json"] is None
    assert len(calls) == 1
