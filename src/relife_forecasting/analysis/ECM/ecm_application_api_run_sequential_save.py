"""
Test client for FastAPI endpoint:
POST /ecm_application/run_sequential_save

- Supports weather_source=pvgis OR epw (multipart upload)
- Prints summary + first few result entries
- Fails loudly on non-2xx responses

Usage:
  python test_run_sequential_save.py

Notes:
- Update BASE_URL if your API runs elsewhere.
- For EPW mode, ensure EPW_PATH exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import requests


BASE_URL = "http://127.0.0.1:9091"
ENDPOINT = "/ecm_application/run_sequential_save"

# --- choose one ---
WEATHER_SOURCE = "epw"   # "pvgis" or "epw"
EPW_PATH = Path("epw_weather/GRC_Athens.167160_IWEC.epw")  # required if WEATHER_SOURCE="epw"

DEFAULT_TIMEOUT = 600


def call_run_sequential_save(
    *,
    archetype: bool = True,
    category: str = "Single Family House",
    country: str = "Greece",
    name: str = "SFH_Greece_1946_1969",
    weather_source: str = "pvgis",
    epw_path: Optional[Path] = None,
    ecm_options: str = "wall,window",
    u_wall: float = 0.5,
    u_roof: Optional[float] = None,
    u_window: float = 1.0,
    include_baseline: bool = True,
    output_dir: str = "results/ecm_api",
    save_bui: bool = True,
    bui_dir: str = "building_examples_ecm_api",
) -> Dict[str, Any]:
    url = f"{BASE_URL}{ENDPOINT}"

    params = {
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
        files = {
            "epw_file": (epw_path.name, epw_path.read_bytes(), "application/octet-stream")
        }

    r = requests.post(url, params=params, files=files, timeout=DEFAULT_TIMEOUT)
    if not r.ok:
        # try to show FastAPI error details
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"HTTP {r.status_code}: {detail}")

    return r.json()


def main() -> None:
    payload = call_run_sequential_save(
        archetype=True,
        category="Single Family House",
        country="Greece",
        name="SFH_Greece_1946_1969",
        weather_source=WEATHER_SOURCE,
        epw_path=EPW_PATH if WEATHER_SOURCE == "epw" else None,
        ecm_options="wall,window",   # combos: baseline + wall + window + wall+window
        u_wall=0.5,
        u_window=1.0,
        u_roof=None,
        include_baseline=True,
        output_dir="results/ecm_api",
        save_bui=True,
        bui_dir="building_examples_ecm_api",
    )

    print("\n=== RESPONSE SUMMARY ===")
    print(json.dumps(payload.get("summary", {}), indent=2, ensure_ascii=False))

    results = payload.get("results", [])
    print(f"\nResults items: {len(results)}")

    # print first few results
    for i, r in enumerate(results[:5], start=1):
        print(f"\n--- #{i} ---")
        print("status:", r.get("status"))
        print("combo_tag:", r.get("combo_tag"))
        if r.get("status") == "success":
            print("files:", r.get("files"))
            print("elapsed_s:", r.get("elapsed_s"))
        else:
            print("error:", r.get("error"))

    # basic assertions
    if payload.get("status") != "completed":
        raise AssertionError("Unexpected status in response")

    if payload.get("summary", {}).get("total", 0) <= 0:
        raise AssertionError("No scenarios were executed (summary.total <= 0)")

    print("\n✅ Test completed OK")


if __name__ == "__main__":
    main()
