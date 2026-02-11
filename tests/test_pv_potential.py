"""
POST /run/iso52016-uni11300-pv

- Sends a BUI dict + PV inputs
- Runs ISO 52016 (PVGIS weather) -> UNI/TS 11300 -> PV+HP matching (PVGIS PV or fallback)
- Asserts response schema + key totals
- Fails loudly on non-2xx responses

Run:
  pytest -q
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import pytest
import requests


BASE_URL = os.getenv("RELIFE_BASE_URL", "http://127.0.0.1:9091")
ENDPOINT = "/run/iso52016-uni11300-pv"
DEFAULT_TIMEOUT = int(os.getenv("RELIFE_TIMEOUT", "600"))

# Optional: pick an archetype via /building, then feed only bui into the pipeline test
DEFAULT_CATEGORY = os.getenv("RELIFE_CATEGORY", "Single Family House")
DEFAULT_COUNTRY = os.getenv("RELIFE_COUNTRY", "Greece")
DEFAULT_ARCHETYPE_NAME = os.getenv("RELIFE_ARCHETYPE_NAME", "SFH_Greece_1946_1969")

# PV defaults (override via env if you want)
PV_KWP = float(os.getenv("RELIFE_PV_KWP", "10"))
PV_TILT = float(os.getenv("RELIFE_PV_TILT", "30"))
PV_AZIMUTH = float(os.getenv("RELIFE_PV_AZIMUTH", "0"))
USE_PVGIS_PV = os.getenv("RELIFE_USE_PVGIS_PV", "true").lower() == "true"
PVGIS_LOSS = float(os.getenv("RELIFE_PVGIS_LOSS", "14"))
PVGIS_YEAR = os.getenv("RELIFE_PVGIS_YEAR", "")  # optional
PVGIS_YEAR = int(PVGIS_YEAR) if PVGIS_YEAR.strip() else None


def _service_available() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _get_bui_from_archetype(
    category: str = DEFAULT_CATEGORY,
    country: str = DEFAULT_COUNTRY,
    name: str = DEFAULT_ARCHETYPE_NAME,
) -> Dict[str, Any]:
    """
    Uses GET /building?archetype=true ... to retrieve archetype BUI and return only 'bui'.
    """
    url = f"{BASE_URL}/building"
    params = {
        "archetype": "true",
        "category": category,
        "country": country,
        "name": name,
    }
    r = requests.post(url, params=params, timeout=60)
    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise AssertionError(f"Failed to fetch archetype BUI: HTTP {r.status_code}: {detail}")

    data = r.json()
    bui = data.get("bui")
    assert isinstance(bui, dict), "Expected 'bui' dict from /building"
    return bui


def call_run_iso52016_uni11300_pv(
    *,
    bui: Dict[str, Any],
    pv_kwp: float = PV_KWP,
    tilt_deg: float = PV_TILT,
    azimuth_deg: float = PV_AZIMUTH,
    use_pvgis: bool = USE_PVGIS_PV,
    pvgis_loss_percent: float = PVGIS_LOSS,
    pvgis_year: int | None = PVGIS_YEAR,
    # optional UNI overrides (keep empty for defaults)
    uni_input_unit: str = "Wh",
    heating_params: Dict[str, Any] | None = None,
    cooling_params: Dict[str, Any] | None = None,
    return_hourly_building: bool = False,
    # optional battery (set env RELIFE_ENABLE_BATTERY=true)
    battery_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    url = f"{BASE_URL}{ENDPOINT}"

    payload: Dict[str, Any] = {
        "bui": bui,
        "pv": {
            "pv_kwp": pv_kwp,
            "tilt_deg": tilt_deg,
            "azimuth_deg": azimuth_deg,
            "use_pvgis": use_pvgis,
            "pvgis_loss_percent": pvgis_loss_percent,
        },
        "uni11300": {
            "input_unit": uni_input_unit,
        },
        "return_hourly_building": return_hourly_building,
    }

    if pvgis_year is not None:
        payload["pv"]["pvgis_year"] = int(pvgis_year)

    if heating_params:
        payload["uni11300"]["heating_params"] = heating_params
    if cooling_params:
        payload["uni11300"]["cooling_params"] = cooling_params

    if battery_params is not None:
        payload["pv"]["battery_params"] = battery_params

    r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise AssertionError(f"HTTP {r.status_code}: {detail}")

    return r.json()


@pytest.mark.parametrize("use_archetype_bui", [True])
def test_run_iso52016_uni11300_pv_endpoint(use_archetype_bui: bool) -> None:
    if not _service_available():
        pytest.skip(f"ReLIFE service not reachable at {BASE_URL}. Start the API to run this test.")

    # --- 1) build BUI input
    if use_archetype_bui:
        bui = _get_bui_from_archetype()
    else:
        # If you want custom mode, inject your bui dict here
        raise pytest.SkipTest("Custom BUI not provided in this test file.")

    # Ensure coords exist (endpoint allows pv overrides, but we keep it simple)
    b = bui.get("building", {})
    assert "latitude" in b and "longitude" in b, "BUI missing building.latitude/longitude (needed for PVGIS/PV)."

    # --- 2) optional battery via env
    enable_battery = os.getenv("RELIFE_ENABLE_BATTERY", "false").lower() == "true"
    battery_params = None
    if enable_battery:
        battery_params = {
            "capacity_kwh": float(os.getenv("RELIFE_BATT_KWH", "10")),
            "max_charge_power_kw": float(os.getenv("RELIFE_BATT_CHG_KW", "5")),
            "max_discharge_power_kw": float(os.getenv("RELIFE_BATT_DIS_KW", "5")),
            "charge_efficiency": float(os.getenv("RELIFE_BATT_CHG_EFF", "0.95")),
            "discharge_efficiency": float(os.getenv("RELIFE_BATT_DIS_EFF", "0.95")),
            "self_discharge_rate_per_hour": float(os.getenv("RELIFE_BATT_SELF_DIS", "0.0001")),
            "min_soc": float(os.getenv("RELIFE_BATT_MIN_SOC", "0.10")),
            "max_soc": float(os.getenv("RELIFE_BATT_MAX_SOC", "0.90")),
            "initial_soc": float(os.getenv("RELIFE_BATT_INIT_SOC", "0.50")),
        }

    # --- 3) call endpoint
    payload = call_run_iso52016_uni11300_pv(
        bui=bui,
        pv_kwp=PV_KWP,
        tilt_deg=PV_TILT,
        azimuth_deg=PV_AZIMUTH,
        use_pvgis=USE_PVGIS_PV,
        pvgis_loss_percent=PVGIS_LOSS,
        pvgis_year=PVGIS_YEAR,
        uni_input_unit="Wh",
        heating_params=None,
        cooling_params=None,
        return_hourly_building=False,
        battery_params=battery_params,
    )

    # --- 4) schema assertions
    assert "inputs" in payload and isinstance(payload["inputs"], dict)
    assert "results" in payload and isinstance(payload["results"], dict)

    results = payload["results"]
    assert "uni11300" in results and isinstance(results["uni11300"], dict)
    assert "pv_hp" in results and isinstance(results["pv_hp"], dict)

    uni = results["uni11300"]
    assert uni.get("ideal_unit") == "kWh"
    assert int(uni.get("n_hours", 0)) >= 8760  # typically 8760 (or 8784 for leap)
    assert "summary" in uni and isinstance(uni["summary"], dict)
    # at least one of these should exist
    assert any(k in uni["summary"] for k in ("EP_total_kWh", "E_delivered_electric_total_kWh", "Q_ideal_heat_kWh"))

    pv_hp = results["pv_hp"]
    assert "summary" in pv_hp and isinstance(pv_hp["summary"], dict)
    assert "hourly_results" in pv_hp and isinstance(pv_hp["hourly_results"], list)
    assert len(pv_hp["hourly_results"]) == uni["n_hours"]

    # --- 5) PV+HP sanity checks
    s = pv_hp["summary"]
    assert "annual_kwh" in s and isinstance(s["annual_kwh"], dict)
    assert "indicators" in s and isinstance(s["indicators"], dict)

    annual = s["annual_kwh"]
    for key in ("hp_electric", "pv_generation", "self_consumption", "grid_import", "grid_export"):
        assert key in annual, f"Missing annual_kwh.{key}"
        assert annual[key] >= 0.0

    ind = s["indicators"]
    assert 0.0 <= ind.get("self_consumption_rate", 0.0) <= 1.0 + 1e-6
    assert 0.0 <= ind.get("self_sufficiency_rate", 0.0) <= 1.0 + 1e-6
