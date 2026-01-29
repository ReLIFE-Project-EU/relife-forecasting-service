"""
ECM API client (SEQUENTIAL) with debug and error handling.

This script:
1) Calls POST /ecm_application (single-scenario mode)
2) Handles EPW upload (weather_source=epw) or PVGIS
3) Saves hourly+annual CSV per scenario (distinct files)
4) (Optional) Saves locally each modified BUI JSON (traceability, only in custom mode unless you fetch /building)
5) Runs calls one after another (no multiprocessing)
"""

from __future__ import annotations

import copy
import itertools
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # fallback


# =============================================================================
# CONFIG
# =============================================================================

BASE_URL = "http://127.0.0.1:9091"
ECM_ENDPOINT = "/ecm_application"

RESULTS_DIR = "results/ecm_api"
BUILDING_EXAMPLES_DIR = "building_examples_ecm_api"

DEFAULT_TIMEOUT = 600  # seconds (ECM can be slow)

# If using archetype mode:
ARCHETYPE = True
CATEGORY = "Single Family House"
COUNTRY = "Greece"
ARCHETYPE_NAME = "SFH_Greece_1946_1969"

# Weather
WEATHER_SOURCE = "epw"  # "pvgis" or "epw"
EPW_PATH = Path("epw_weather/GRC_Athens.167160_IWEC.epw")  # change to your path if needed

# ECM controls
ECM_OPTIONS = ["wall", "window"]  # subset of ["roof","wall","window"]
U_WALL = 0.5
U_ROOF = None
U_WINDOW = 1.0

# baseline included?
INCLUDE_BASELINE = True


# =============================================================================
# UTILITIES
# =============================================================================

def slugify(text: str) -> str:
    return (
        str(text).lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def ensure_dir(p: Union[str, Path]) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def json_safe(obj: Any) -> Any:
    """Convert numpy types into JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    return obj


def save_bui_to_folder(bui_obj: Dict[str, Any], active_elements: List[str], folder: str = BUILDING_EXAMPLES_DIR) -> str:
    ensure_dir(folder)
    building_name = bui_obj.get("building", {}).get("name", "building")
    combo_tag = "_".join(sorted(active_elements)) if active_elements else "baseline"

    filename = f"BUI_{slugify(building_name)}__{combo_tag}.json"
    full_path = Path(folder) / filename

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(bui_obj), f, ensure_ascii=False, indent=2)

    return str(full_path)


def build_name_file(
    building_name: str,
    ecm_combo: List[str],
    epw_path: Optional[Union[str, Path]],
    base_dir: Union[str, Path],
) -> str:
    base_dir = ensure_dir(base_dir)
    bld = slugify(building_name)
    ecm_tag = "_".join(sorted(ecm_combo)) if ecm_combo else "baseline"
    weather_tag = "pvgis"

    if epw_path:
        epw = Path(epw_path)
        weather_tag = slugify(epw.stem)

    filename = f"{bld}__{ecm_tag}__{weather_tag}.csv"
    return str(base_dir / filename)


def to_dataframe(payload: Any) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    if isinstance(payload, dict):
        try:
            return pd.DataFrame.from_dict(payload)
        except Exception:
            return pd.DataFrame([payload])

    return pd.DataFrame([{"value": payload}])


def generate_ecm_combinations(ecm_options: List[str], include_baseline: bool = True) -> List[List[str]]:
    combos: List[List[str]] = []
    if include_baseline:
        combos.append([])

    for r in range(1, len(ecm_options) + 1):
        for subset in itertools.combinations(ecm_options, r):
            combos.append(list(subset))

    return combos


# =============================================================================
# LOCAL "APPLY U" (for saving BUI variants) - only meaningful in custom mode unless you fetch /building first.
# =============================================================================

def classify_surface(surface: Dict[str, Any]) -> Optional[str]:
    s_type = str(surface.get("type", "")).lower()
    ori = surface.get("orientation", {})
    tilt = float(ori.get("tilt", 0))
    azimuth = float(ori.get("azimuth", 0))
    name = str(surface.get("name", "")).lower()
    svf = float(surface.get("sky_view_factor", 0.0))

    if (
        s_type == "opaque"
        and abs(tilt - 0) < 1e-3
        and abs(azimuth - 0) < 1e-3
        and svf > 0.01
        and "slab" not in name
    ):
        return "roof"

    if s_type == "opaque" and abs(tilt - 90) < 1e-3:
        return "wall"

    if s_type == "transparent" and abs(tilt - 90) < 1e-3:
        return "window"

    return None


def apply_u_values_to_BUI(
    bui_base: Dict[str, Any],
    active_elements: List[str],
    u_wall: Optional[float],
    u_roof: Optional[float],
    u_window: Optional[float],
) -> Dict[str, Any]:
    u_map: Dict[str, float] = {}
    if "wall" in active_elements and u_wall is not None:
        u_map["wall"] = float(u_wall)
    if "roof" in active_elements and u_roof is not None:
        u_map["roof"] = float(u_roof)
    if "window" in active_elements and u_window is not None:
        u_map["window"] = float(u_window)

    bui_new = copy.deepcopy(bui_base)

    for s in bui_new.get("building_surface", []):
        c = classify_surface(s)
        if c in u_map:
            s["u_value"] = u_map[c]

    save_bui_to_folder(bui_new, active_elements=active_elements)
    return bui_new


# =============================================================================
# API CALLS
# =============================================================================

@dataclass
class ApiTask:
    combo: List[str]
    u_wall: Optional[float]
    u_roof: Optional[float]
    u_window: Optional[float]
    weather_source: str
    epw_path: Optional[str]
    output_dir: str

    # archetype mode
    archetype: bool
    category: Optional[str]
    country: Optional[str]
    name: Optional[str]

    # custom mode (optional)
    bui_json: Optional[Dict[str, Any]] = None

    # single scenario mode params
    scenario_elements: Optional[str] = None
    baseline_only: bool = False


def call_ecm_application(task: ApiTask) -> Dict[str, Any]:
    """
    Calls /ecm_application in SINGLE SCENARIO MODE (sequential-friendly):
      - baseline_only=true -> baseline only
      - scenario_elements=wall,window -> only that scenario
    """
    start = time.time()
    combo = task.combo

    params: Dict[str, Any] = {
        "archetype": str(task.archetype).lower(),
        "weather_source": task.weather_source,
        "u_wall": task.u_wall,
        "u_roof": task.u_roof,
        "u_window": task.u_window,
    }

    # single scenario selector
    if task.baseline_only:
        params["baseline_only"] = "true"
    else:
        params["scenario_elements"] = task.scenario_elements or (",".join(combo) if combo else "")
        if not params["scenario_elements"]:
            params["baseline_only"] = "true"

    if task.archetype:
        params.update({"category": task.category, "country": task.country, "name": task.name})

    data: Dict[str, Any] = {}
    files: Dict[str, Any] = {}

    if not task.archetype:
        if task.bui_json is None:
            raise ValueError("Custom mode requires bui_json")
        data["bui_json"] = json.dumps(task.bui_json)

    if task.weather_source == "epw":
        if not task.epw_path:
            raise ValueError("weather_source='epw' requires epw_path")
        epw_p = Path(task.epw_path)
        files["epw_file"] = (epw_p.name, epw_p.read_bytes(), "application/octet-stream")

    url = f"{BASE_URL}{ECM_ENDPOINT}"

    try:
        r = requests.post(url, params=params, data=data, files=files if files else None, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        payload = r.json()

        scenarios = payload.get("scenarios", [])
        if not isinstance(scenarios, list) or len(scenarios) == 0:
            raise ValueError("API response has no 'scenarios'")

        # Single scenario mode => take last (safe if server returns [baseline, scenario] unexpectedly)
        chosen = scenarios[-1]

        building_name = task.name or "custom_building"
        hourly = to_dataframe(chosen.get("results", {}).get("hourly_building"))
        annual = to_dataframe(chosen.get("results", {}).get("annual_building"))

        name_file = build_name_file(
            building_name=building_name,
            ecm_combo=combo,
            epw_path=task.epw_path if task.weather_source == "epw" else None,
            base_dir=task.output_dir,
        )
        hourly.to_csv(name_file, index=False)

        annual_file = str(Path(name_file).with_name(Path(name_file).stem + "__annual.csv"))
        annual.to_csv(annual_file, index=False)

        elapsed = time.time() - start
        size_kb = Path(name_file).stat().st_size / 1024.0 if Path(name_file).exists() else 0.0

        return {
            "status": "success",
            "combo": combo,
            "scenario_id": chosen.get("scenario_id"),
            "file_hourly": name_file,
            "file_annual": annual_file,
            "elapsed": elapsed,
            "size_kb": size_kb,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": "error",
            "combo": combo,
            "elapsed": elapsed,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# MAIN ORCHESTRATION (SEQUENTIAL)
# =============================================================================

def run_ecm_api_sequential(
    ecm_options: List[str],
    u_wall: Optional[float],
    u_roof: Optional[float],
    u_window: Optional[float],
    weather_source: str,
    epw_path: Optional[Union[str, Path]],
    output_dir: Union[str, Path],
    include_baseline: bool = True,
    archetype: bool = True,
    category: Optional[str] = None,
    country: Optional[str] = None,
    name: Optional[str] = None,
    bui_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # validations
    if "wall" in ecm_options and u_wall is None:
        raise ValueError("For ECM 'wall' you must provide u_wall.")
    if "roof" in ecm_options and u_roof is None:
        raise ValueError("For ECM 'roof' you must provide u_roof.")
    if "window" in ecm_options and u_window is None:
        raise ValueError("For ECM 'window' you must provide u_window.")

    ensure_dir(output_dir)
    ensure_dir(BUILDING_EXAMPLES_DIR)

    combos = generate_ecm_combinations(ecm_options, include_baseline=include_baseline)

    # Build tasks (one after another)
    tasks: List[ApiTask] = []
    for combo in combos:
        is_baseline = (len(combo) == 0)

        # Optional: save local BUI variants (ONLY works if you have bui_json, i.e. custom mode)
        if not archetype and bui_json is not None:
            _ = apply_u_values_to_BUI(
                bui_base=bui_json,
                active_elements=combo,
                u_wall=u_wall,
                u_roof=u_roof,
                u_window=u_window,
            )

        tasks.append(
            ApiTask(
                combo=combo,
                u_wall=u_wall if ("wall" in ecm_options) else None,
                u_roof=u_roof if ("roof" in ecm_options) else None,
                u_window=u_window if ("window" in ecm_options) else None,
                weather_source=weather_source,
                epw_path=str(epw_path) if epw_path else None,
                output_dir=str(output_dir),
                archetype=archetype,
                category=category,
                country=country,
                name=name,
                bui_json=bui_json,
                scenario_elements=None if is_baseline else ",".join(combo),
                baseline_only=is_baseline,
            )
        )

    start = time.time()
    results: List[Dict[str, Any]] = []

    iterator = tasks
    if tqdm:
        iterator = tqdm(tasks, desc="ECM API (sequential)", total=len(tasks))

    for t in iterator:
        results.append(call_ecm_application(t))

    total_time = time.time() - start
    ok = [r for r in results if r["status"] == "success"]
    ko = [r for r in results if r["status"] == "error"]

    return {
        "total": len(results),
        "successful": len(ok),
        "failed": len(ko),
        "total_time": total_time,
        "results": results,
    }


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    stats = run_ecm_api_sequential(
        ecm_options=ECM_OPTIONS,
        u_wall=U_WALL,
        u_roof=U_ROOF,
        u_window=U_WINDOW,
        weather_source=WEATHER_SOURCE,
        epw_path=EPW_PATH if WEATHER_SOURCE == "epw" else None,
        output_dir=RESULTS_DIR,
        include_baseline=INCLUDE_BASELINE,
        archetype=ARCHETYPE,
        category=CATEGORY,
        country=COUNTRY,
        name=ARCHETYPE_NAME,
    )

    print("\n" + "=" * 80)
    print("COMPLETED (ECM API - SEQUENTIAL)")
    print("=" * 80)
    print(f"Successes: {stats['successful']}/{stats['total']}")
    print(f"Failed: {stats['failed']}/{stats['total']}")
    print(f"Time: {stats['total_time']:.1f}s")

    if stats["failed"]:
        print("\n[ERRORS]")
        for r in stats["results"]:
            if r["status"] == "error":
                print(f"- combo={r.get('combo')}: {r.get('error')}")
