# main_call_pipeline.py
# -----------------------------------------------------------------------------
# Small "main" script to call:
#   POST /run/iso52016-uni11300-pv
#
# Usage:
#   python main_call_pipeline.py
#
# Env overrides:
#   RELIFE_BASE_URL=http://127.0.0.1:9091
#   RELIFE_TIMEOUT=600
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests

# Prefer package import; fallback to local module for dev runs.
try:
    from relife_forecasting.building_examples import BUI_SINGLE_FAMILY_1946_1969
except Exception:
    from building_examples import BUI_SINGLE_FAMILY_1946_1969  # type: ignore


BASE_URL = os.getenv("RELIFE_BASE_URL", "http://127.0.0.1:9091")
ENDPOINT = "/run/iso52016-uni11300-pv"
TIMEOUT = int(os.getenv("RELIFE_TIMEOUT", "600"))


# -----------------------------------------------------------------------------
# PV inputs: keep them local to this script so we don't depend on extra constants
# -----------------------------------------------------------------------------
PV_INPUTS_DEFAULT: Dict[str, Any] = {
    "pv_kwp": 10.0,
    "tilt_deg": 30.0,
    "azimuth_deg": 0.0,  # convention in your pv_hp_analysis: 0=south, 90=west, -90=east
    "use_pvgis": True,
    "pvgis_loss_percent": 14.0,
    # "pvgis_year": 2019,
    "annual_pv_yield_kwh_per_kwp": 1400.0,  # fallback only
    "battery_params": {
        "capacity_kwh": 10.0,
        "max_charge_power_kw": 5.0,
        "max_discharge_power_kw": 5.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "self_discharge_rate_per_hour": 0.0001,
        "min_soc": 0.10,
        "max_soc": 0.90,
        "initial_soc": 0.50,
    },
}


def _fail_loud(resp: requests.Response) -> None:
    if resp.ok:
        return
    try:
        detail = resp.json()
    except Exception:
        detail = resp.text
    raise SystemExit(f"HTTP {resp.status_code} calling {resp.url}: {detail}")


def _get_health_url() -> str:
    # In main.py you included_router(health.router) (usually /health)
    return f"{BASE_URL}/health"


def call_pipeline(
    bui: Dict[str, Any],
    pv: Dict[str, Any],
    *,
    input_unit: str = "Wh",
    return_hourly_building: bool = False,
    heating_params: Optional[Dict[str, Any]] = None,
    cooling_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = f"{BASE_URL}{ENDPOINT}"

    payload: Dict[str, Any] = {
        "bui": bui,
        "pv": pv,
        "uni11300": {
            "input_unit": input_unit,
        },
        "return_hourly_building": return_hourly_building,
    }
    if heating_params:
        payload["uni11300"]["heating_params"] = heating_params
    if cooling_params:
        payload["uni11300"]["cooling_params"] = cooling_params

    resp = requests.post(url, json=payload, timeout=TIMEOUT)
    _fail_loud(resp)
    return resp.json()


def main() -> None:
    # quick availability check
    try:
        h = requests.get(_get_health_url(), timeout=5)
        if h.status_code != 200:
            raise SystemExit(f"Service not healthy: {h.status_code} {h.text}")
    except requests.RequestException as exc:
        raise SystemExit(f"Service not reachable at {BASE_URL}: {exc}") from exc

    # Note: endpoint allows pv.latitude/pv.longitude override, otherwise uses bui.building coords.
    # So PV_INPUTS_DEFAULT doesn't need coordinates unless you want to override them.
    result = call_pipeline(
        bui=BUI_SINGLE_FAMILY_1946_1969,
        pv=PV_INPUTS_DEFAULT,
        input_unit="Wh",
        return_hourly_building=False,
    )

    # print compact summary
    uni = result.get("results", {}).get("uni11300", {})
    pv_hp = result.get("results", {}).get("pv_hp", {})

    print("\n=== UNI/TS 11300 summary ===")
    print(json.dumps(uni.get("summary", {}), indent=2, ensure_ascii=False))

    print("\n=== PV+HP annual_kwh ===")
    print(json.dumps(pv_hp.get("summary", {}).get("annual_kwh", {}), indent=2, ensure_ascii=False))

    print("\n=== PV+HP indicators ===")
    print(json.dumps(pv_hp.get("summary", {}).get("indicators", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
