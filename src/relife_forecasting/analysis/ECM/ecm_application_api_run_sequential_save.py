"""
Test client for FastAPI endpoint:
POST /ecm_application/run_sequential_save

- Supports weather_source=pvgis OR epw (multipart upload)
- Generates and saves the HTML comparison report from /ecm_application/report
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
REPORT_ENDPOINT = "/ecm_application/report"

# --- choose one ---
WEATHER_SOURCE = "epw"   # "pvgis" or "epw"
EPW_PATH = Path("epw_weather/GRC_Athens.167160_IWEC.epw")  # required if WEATHER_SOURCE="epw"

DEFAULT_TIMEOUT = 600
GENERATE_REPORT = True


def slugify(text: str) -> str:
    return (
        str(text).lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def ensure_dir(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def build_report_path(
    *,
    building_name: str,
    output_dir: str,
    epw_path: Optional[Path],
) -> Path:
    weather_tag = "pvgis"
    if epw_path is not None:
        weather_tag = slugify(epw_path.stem)
    filename = f"{slugify(building_name)}__ecm_report__{weather_tag}.html"
    return ensure_dir(output_dir) / filename


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


def call_ecm_report(
    *,
    archetype: bool = True,
    category: str = "Single Family House",
    country: str = "Greece",
    name: str = "SFH_Greece_1946_1969",
    weather_source: str = "pvgis",
    epw_path: Optional[Path] = None,
    scenario_elements: Optional[str] = None,
    u_wall: Optional[float] = None,
    u_roof: Optional[float] = None,
    u_window: Optional[float] = None,
    u_slab: Optional[float] = None,
    report_title: Optional[str] = None,
) -> str:
    url = f"{BASE_URL}{REPORT_ENDPOINT}"

    params: Dict[str, Any] = {
        "archetype": str(archetype).lower(),
        "category": category,
        "country": country,
        "name": name,
        "weather_source": weather_source,
    }
    if scenario_elements:
        params["scenario_elements"] = scenario_elements
    if u_wall is not None:
        params["u_wall"] = u_wall
    if u_roof is not None:
        params["u_roof"] = u_roof
    if u_window is not None:
        params["u_window"] = u_window
    if u_slab is not None:
        params["u_slab"] = u_slab
    if report_title:
        params["report_title"] = report_title

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
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"HTTP {r.status_code}: {detail}")

    return r.text


def main() -> None:
    ecm_options = "wall,window"
    payload = call_run_sequential_save(
        archetype=True,
        category="Single Family House",
        country="Greece",
        name="SFH_Greece_1946_1969",
        weather_source=WEATHER_SOURCE,
        epw_path=EPW_PATH if WEATHER_SOURCE == "epw" else None,
        ecm_options=ecm_options,   # combos: baseline + wall + window + wall+window
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

    if GENERATE_REPORT:
        report_elements = [
            option.strip().lower()
            for option in ecm_options.split(",")
            if option.strip() and option.strip().lower() in {"wall", "roof", "window", "slab"}
        ]
        if report_elements:
            report_html = call_ecm_report(
                archetype=True,
                category="Single Family House",
                country="Greece",
                name="SFH_Greece_1946_1969",
                weather_source=WEATHER_SOURCE,
                epw_path=EPW_PATH if WEATHER_SOURCE == "epw" else None,
                scenario_elements=",".join(report_elements),
                u_wall=0.5,
                u_window=1.0,
                u_roof=None,
                report_title="ECM comparison report",
            )
            report_path = build_report_path(
                building_name="SFH_Greece_1946_1969",
                output_dir="results/ecm_api",
                epw_path=EPW_PATH if WEATHER_SOURCE == "epw" else None,
            )
            report_path.write_text(report_html, encoding="utf-8")
            print(f"\nHTML report saved to: {report_path}")
        else:
            print("\nHTML report skipped: no envelope ECM selected for comparison.")

    print("\n✅ Test completed OK")


if __name__ == "__main__":
    main()
