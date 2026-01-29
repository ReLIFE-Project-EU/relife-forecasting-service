"""
POST /ecm_application/run_sequential_save

- Supports weather_source=pvgis OR epw (multipart upload)
- Asserts response schema + that files were created (paths returned)
- Fails loudly on non-2xx responses

Run:
  pytest -q
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any

import pytest
import requests


BASE_URL = os.getenv("RELIFE_BASE_URL", "http://127.0.0.1:9091")
ENDPOINT = "/ecm_application/run_sequential_save"
DEFAULT_TIMEOUT = int(os.getenv("RELIFE_TIMEOUT", "600"))

# Default test config (override via env if you want)
DEFAULT_CATEGORY = os.getenv("RELIFE_CATEGORY", "Single Family House")
DEFAULT_COUNTRY = os.getenv("RELIFE_COUNTRY", "Greece")
DEFAULT_ARCHETYPE_NAME = os.getenv("RELIFE_ARCHETYPE_NAME", "SFH_Greece_1946_1969")

# Weather selection:
# RELIFE_WEATHER_SOURCE=pvgis  (no file needed)
# RELIFE_WEATHER_SOURCE=epw    (needs RELIFE_EPW_PATH)
WEATHER_SOURCE = os.getenv("RELIFE_WEATHER_SOURCE", "pvgis").lower()
EPW_PATH = Path(os.getenv("RELIFE_EPW_PATH", "epw_weather/GRC_Athens.167160_IWEC.epw"))

# Where endpoint will save
OUTPUT_DIR = os.getenv("RELIFE_OUTPUT_DIR", "results/ecm_api")
BUI_DIR = os.getenv("RELIFE_BUI_DIR", "building_examples_ecm_api")


def _service_available() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def call_run_sequential_save(
    *,
    archetype: bool = True,
    category: str = DEFAULT_CATEGORY,
    country: str = DEFAULT_COUNTRY,
    name: str = DEFAULT_ARCHETYPE_NAME,
    weather_source: str = "pvgis",
    epw_path: Optional[Path] = None,
    ecm_options: str = "wall,window",
    u_wall: float = 0.5,
    u_roof: Optional[float] = None,
    u_window: float = 1.0,
    include_baseline: bool = True,
    output_dir: str = OUTPUT_DIR,
    save_bui: bool = True,
    bui_dir: str = BUI_DIR,
) -> Dict[str, Any]:
    url = f"{BASE_URL}{ENDPOINT}"

    params: Dict[str, Any] = {
        "archetype": str(archetype).lower(),
        "category": category,
        "country": country,
        "name": name,
        "weather_source": weather_source,
        "ecm_options": ecm_options,
        "u_wall": u_wall,
        "u_window": u_window,
        "include_baseline": str(include_baseline).lower(),
        "output_dir": output_dir,
        "save_bui": str(save_bui).lower(),
        "bui_dir": bui_dir,
    }
    if u_roof is not None:
        params["u_roof"] = u_roof

    files = None
    if weather_source == "epw":
        if epw_path is None:
            raise ValueError("epw_path is required when weather_source='epw'")
        if not epw_path.exists():
            raise FileNotFoundError(f"EPW file not found: {epw_path}")
        files = {"epw_file": (epw_path.name, epw_path.read_bytes(), "application/octet-stream")}

    r = requests.post(url, params=params, files=files, timeout=DEFAULT_TIMEOUT)

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise AssertionError(f"HTTP {r.status_code}: {detail}")

    return r.json()


@pytest.mark.parametrize("weather_source", [WEATHER_SOURCE])
def test_run_sequential_save_endpoint(weather_source: str) -> None:
    if not _service_available():
        pytest.skip(f"ReLIFE service not reachable at {BASE_URL}. Start the API to run this test.")

    # If EPW requested but file missing -> fallback to PVGIS
    if weather_source == "epw" and not EPW_PATH.exists():
        weather_source = "pvgis"

    payload = call_run_sequential_save(
        archetype=True,
        category=DEFAULT_CATEGORY,
        country=DEFAULT_COUNTRY,
        name=DEFAULT_ARCHETYPE_NAME,
        weather_source=weather_source,
        epw_path=EPW_PATH if weather_source == "epw" else None,
        ecm_options="wall,window",
        u_wall=0.5,
        u_window=1.0,
        u_roof=None,
        include_baseline=True,
        output_dir=OUTPUT_DIR,
        save_bui=True,
        bui_dir=BUI_DIR,
    )

    # ---- basic schema assertions
    assert payload.get("status") == "completed"
    summary = payload.get("summary") or {}
    assert summary.get("total", 0) > 0
    assert summary.get("successful", 0) + summary.get("failed", 0) == summary.get("total", 0)

    results = payload.get("results")
    assert isinstance(results, list)
    assert len(results) == summary["total"]

    # ---- per-result assertions + file existence on success
    # NOTE: files are saved on SERVER filesystem; this test assumes it runs on same machine.
    for item in results:
        assert item.get("status") in {"success", "error"}
        assert "combo" in item

        if item["status"] == "success":
            files = item.get("files") or {}
            hourly_csv = files.get("hourly_csv")
            annual_csv = files.get("annual_csv")

            assert hourly_csv and isinstance(hourly_csv, str)
            assert annual_csv and isinstance(annual_csv, str)

            assert Path(hourly_csv).exists(), f"hourly_csv missing on disk: {hourly_csv}"
            assert Path(annual_csv).exists(), f"annual_csv missing on disk: {annual_csv}"

            # bui json might be None if save_bui=false
            bui_json = files.get("bui_json")
            if bui_json is not None:
                assert Path(bui_json).exists(), f"bui_json missing on disk: {bui_json}"
        else:
            assert item.get("error"), "Error status without error message"
