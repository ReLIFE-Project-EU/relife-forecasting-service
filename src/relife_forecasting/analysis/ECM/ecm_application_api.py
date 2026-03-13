"""
ECM API client with debug and error handling.

This script:
1) Calls POST /ecm_application for baseline
2) Calls POST /ecm_application for the full ECM scenario (all selected ECM together)
3) Generates the HTML comparison report locally from the two results already received
4) Handles EPW upload (weather_source=epw) or PVGIS
5) Saves hourly+annual CSV per scenario (distinct files)
6) (Optional) Saves locally each modified BUI JSON (traceability, only in custom mode unless you fetch /building)
7) Runs calls one after another (no multiprocessing)
"""

from __future__ import annotations

import copy
import importlib
import itertools
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests


def _ensure_project_src_on_path() -> None:
    candidates: List[Path] = []

    file_path = globals().get("__file__")
    if file_path:
        candidates.append(Path(file_path).resolve())
    candidates.append(Path.cwd().resolve())

    src_paths: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            for src_dir in (parent / "src", parent):
                package_dir = src_dir / "relife_forecasting"
                src_key = str(src_dir)
                if package_dir.is_dir() and src_key not in seen:
                    seen.add(src_key)
                    src_paths.append(src_dir)

    for src_dir in reversed(src_paths):
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_project_src_on_path()


def _load_report_builders():
    last_exc: Optional[Exception] = None
    for module_name in (
        "relife_forecasting.utils.ecm_report_html",
        "utils.ecm_report_html",
    ):
        try:
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
            return (
                module.build_ecm_comparison_report_html,
                module.build_ecm_multi_scenario_report_html,
            )
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ImportError("Unable to import ECM report builders.")


(
    build_ecm_comparison_report_html,
    build_ecm_multi_scenario_report_html,
) = _load_report_builders()

try:
    from relife_forecasting.building_examples import BUILDING_ARCHETYPES, UNI11300_SIMULATION_EXAMPLE
except Exception:
    from building_examples import BUILDING_ARCHETYPES, UNI11300_SIMULATION_EXAMPLE  # type: ignore

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
REPORT_ENDPOINT = "/ecm_application/report"
UNI_PV_ENDPOINT = "/run/iso52016-uni11300-pv"

RESULTS_DIR = "results/ecm_api"
BUILDING_EXAMPLES_DIR = "building_examples_ecm_api"

DEFAULT_TIMEOUT = 600  # seconds (ECM can be slow)

# If using archetype mode:
ARCHETYPE = True
CATEGORY = "Single Family House"
COUNTRY = "Greece"
ARCHETYPE_NAME = "SFH_Greece_1946_1969"

# Weather
WEATHER_SOURCE = "pvgis"  # "pvgis" or "epw"; keep "pvgis" when using the scenario library with PV
EPW_PATH = Path("epw_weather/GRC_Athens.167160_IWEC.epw")  # change to your path if needed

# ECM controls
ECM_OPTIONS = ["wall", "window", "roof", "slab", "heat_pump"]  # subset of ["roof","wall","window","slab","heat_pump"]
U_WALL = 0.5
U_ROOF = 0.3
U_WINDOW = 1.0
U_FLOOR = 0.4  # slab-to-ground (floor) U-value
U_SLAB = U_FLOOR  # alias for API parameter name

# Generation update (optional)
USE_HEAT_PUMP = True
HEAT_PUMP_COP_DEFAULT = 3.2
# Optionally override COP per combo tag (e.g. "baseline", "wall", "wall_window", etc.)
HEAT_PUMP_COP_BY_COMBO: Dict[str, float] = {}

# baseline included?
INCLUDE_BASELINE = True
GENERATE_REPORT = True
REPORT_TITLE = "ECM comparison report"
USE_RENOVATION_SCENARIO_LIBRARY = True
# Leave empty to compare all selected renovation scenarios in one report.
REPORT_SCENARIO_NAME: Optional[str] = None

# PV defaults
PV_KWP = 6.0
PV_TILT_DEG = 30.0
PV_AZIMUTH_DEG = 0.0
PV_USE_PVGIS = True
PV_PVGIS_LOSS_PERCENT = 14.0
ANNUAL_PV_YIELD_KWH_PER_KWP = 1400.0


@dataclass(frozen=True)
class RenovationScenario:
    name: str
    ecm_options: List[str]
    description: str
    use_heat_pump: bool = False
    heat_pump_cop: float = HEAT_PUMP_COP_DEFAULT
    uni_generation_mode: str = "default"
    uni_eta_generation: Optional[float] = None
    use_pv: bool = False
    pv_config: Optional[Dict[str, Any]] = None


DEFAULT_PV_CONFIG: Dict[str, Any] = {
    "pv_kwp": PV_KWP,
    "tilt_deg": PV_TILT_DEG,
    "azimuth_deg": PV_AZIMUTH_DEG,
    "use_pvgis": PV_USE_PVGIS,
    "pvgis_loss_percent": PV_PVGIS_LOSS_PERCENT,
    "annual_pv_yield_kwh_per_kwp": ANNUAL_PV_YIELD_KWH_PER_KWP,
}


RENOVATION_SCENARIOS: List[RenovationScenario] = [
    # RenovationScenario(
    #     name="wall_insulation",
    #     description="External wall insulation",
    #     ecm_options=["wall"],
    # ),
    # RenovationScenario(
    #     name="window_replacement",
    #     description="High-performance window replacement",
    #     ecm_options=["window"],
    # ),
    # RenovationScenario(
    #     name="roof_insulation",
    #     description="Roof insulation retrofit",
    #     ecm_options=["roof"],
    # ),
    # RenovationScenario(
    #     name="floor_insulation",
    #     description="Slab or floor insulation retrofit",
    #     ecm_options=["slab"],
    # ),
    RenovationScenario(
        name="deep_envelope",
        description="Deep envelope retrofit",
        ecm_options=["wall", "window", "roof", "slab"],
    ),
    RenovationScenario(
        name="condensing_boiler",
        description="Condensing boiler replacement",
        ecm_options=[],
        uni_generation_mode="condensing_boiler",
    ),
    # RenovationScenario(
    #     name="deep_envelope_condensing_boiler",
    #     description="Deep envelope retrofit with condensing boiler",
    #     ecm_options=["wall", "window", "roof", "slab"],
    #     uni_generation_mode="condensing_boiler",
    # ),
    # RenovationScenario(
    #     name="heat_pump",
    #     description="Heat pump retrofit",
    #     ecm_options=["heat_pump"],
    #     use_heat_pump=True,
    #     heat_pump_cop=HEAT_PUMP_COP_DEFAULT,
    # ),
    # RenovationScenario(
    #     name="pv_only",
    #     description="Photovoltaic system installation",
    #     ecm_options=[],
    #     use_pv=True,
    #     pv_config=DEFAULT_PV_CONFIG,
    # ),
    # RenovationScenario(
    #     name="pv_condensing_boiler",
    #     description="Photovoltaic system with condensing boiler",
    #     ecm_options=[],
    #     use_pv=True,
    #     pv_config=DEFAULT_PV_CONFIG,
    #     uni_generation_mode="condensing_boiler",
    # ),
    # RenovationScenario(
    #     name="deep_retrofit_hp_pv",
    #     description="Deep envelope retrofit with heat pump and photovoltaics",
    #     ecm_options=["wall", "window", "roof", "slab", "heat_pump"],
    #     use_heat_pump=True,
    #     heat_pump_cop=HEAT_PUMP_COP_DEFAULT,
    #     use_pv=True,
    #     pv_config=DEFAULT_PV_CONFIG,
    # ),
]


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
    scenario_name: Optional[str] = None,
) -> str:
    base_dir = ensure_dir(base_dir)
    bld = slugify(building_name)
    ecm_tag = slugify(scenario_name) if scenario_name else ("_".join(sorted(ecm_combo)) if ecm_combo else "baseline")
    weather_tag = "pvgis"

    if epw_path:
        epw = Path(epw_path)
        weather_tag = slugify(epw.stem)

    filename = f"{bld}__{ecm_tag}__{weather_tag}.csv"
    return str(base_dir / filename)


def build_report_file(
    building_name: str,
    ecm_combo: List[str],
    epw_path: Optional[Union[str, Path]],
    base_dir: Union[str, Path],
    scenario_name: Optional[str] = None,
) -> str:
    base_dir = ensure_dir(base_dir)
    bld = slugify(building_name)
    ecm_tag = slugify(scenario_name) if scenario_name else ("_".join(sorted(ecm_combo)) if ecm_combo else "baseline")
    weather_tag = "pvgis"

    if epw_path:
        epw = Path(epw_path)
        weather_tag = slugify(epw.stem)

    filename = f"{bld}__{ecm_tag}__report__{weather_tag}.html"
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


def extract_envelope_elements(options: List[str]) -> List[str]:
    return [option for option in options if option in {"wall", "roof", "window", "slab"}]


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

    if s_type == "opaque" and abs(tilt - 0) < 1e-3 and abs(azimuth - 0) < 1e-3:
        if ("slab" in name) or ("ground" in name) or (svf <= 0.01):
            return "slab"
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
    u_slab: Optional[float],
) -> Dict[str, Any]:
    u_map: Dict[str, float] = {}
    if "wall" in active_elements and u_wall is not None:
        u_map["wall"] = float(u_wall)
    if "roof" in active_elements and u_roof is not None:
        u_map["roof"] = float(u_roof)
    if "window" in active_elements and u_window is not None:
        u_map["window"] = float(u_window)
    if "slab" in active_elements and u_slab is not None:
        u_map["slab"] = float(u_slab)

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
    scenario_name: Optional[str]
    combo: List[str]
    u_wall: Optional[float]
    u_roof: Optional[float]
    u_window: Optional[float]
    u_slab: Optional[float]
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
    include_baseline: bool = False

    # generation update (optional)
    use_heat_pump: bool = False
    heat_pump_cop: float = 3.2
    uni_generation_mode: str = "default"
    uni_eta_generation: Optional[float] = None
    pv_config: Optional[Dict[str, Any]] = None
    scenario_description: Optional[str] = None


def call_ecm_application(task: ApiTask) -> Dict[str, Any]:
    """
    Calls /ecm_application in SINGLE SCENARIO MODE (sequential-friendly):
      - baseline_only=true -> baseline only
      - scenario_elements=wall,window -> only that scenario
      - scenario without scenario_elements -> generation-only ECM
    """
    start = time.time()
    combo = task.combo
    combo_tag = "_".join(sorted(combo)) if combo else "baseline"
    scenario_name = task.scenario_name or combo_tag

    params: Dict[str, Any] = {
        "archetype": str(task.archetype).lower(),
        "weather_source": task.weather_source,
        "u_wall": task.u_wall,
        "u_roof": task.u_roof,
        "u_window": task.u_window,
        "u_slab": task.u_slab,
    }
    if task.use_heat_pump:
        params["use_heat_pump"] = "true"
        params["heat_pump_cop"] = task.heat_pump_cop
    if str(task.uni_generation_mode or "default").strip().lower() != "default":
        params["uni_generation_mode"] = task.uni_generation_mode
    if task.uni_eta_generation is not None:
        params["uni_eta_generation"] = task.uni_eta_generation
    if task.include_baseline:
        params["include_baseline"] = "true"

    # single scenario selector
    envelope_combo = extract_envelope_elements(combo)
    if task.baseline_only:
        params["baseline_only"] = "true"
    else:
        scenario_elements = task.scenario_elements or (",".join(envelope_combo) if envelope_combo else "")
        if scenario_elements:
            params["scenario_elements"] = scenario_elements

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
        chosen_results = chosen.get("results", {}) if isinstance(chosen, dict) else {}
        chosen_uni_results = chosen_results.get("primary_energy_uni11300", {}) if isinstance(chosen_results, dict) else {}

        task_building = task.bui_json.get("building", {}) if isinstance(task.bui_json, dict) else {}
        building_name = (
            payload.get("name")
            or task.name
            or task_building.get("name")
            or "custom_building"
        )
        building_category = (
            payload.get("category")
            or task.category
            or task_building.get("building_type_class")
        )
        building_country = payload.get("country") or task.country
        generation_mask = (
            chosen_uni_results.get("generation_mask")
            if isinstance(chosen_uni_results, dict)
            else None
        ) or payload.get("uni11300_generation_mask") or {}
        scenario_label = (
            scenario_name
            if scenario_name != "baseline"
            else (chosen.get("description") or chosen.get("scenario_id") or combo_tag)
        )
        scenario_description = (
            task.scenario_description
            or chosen.get("description")
            or scenario_label
        )

        hourly = to_dataframe(chosen_results.get("hourly_building"))
        annual = to_dataframe(chosen_results.get("annual_building"))

        name_file = build_name_file(
            building_name=building_name,
            ecm_combo=combo,
            epw_path=task.epw_path if task.weather_source == "epw" else None,
            base_dir=task.output_dir,
            scenario_name=scenario_name,
        )
        hourly.to_csv(name_file, index=False)

        annual_file = str(Path(name_file).with_name(Path(name_file).stem + "__annual.csv"))
        annual.to_csv(annual_file, index=False)

        elapsed = time.time() - start
        size_kb = Path(name_file).stat().st_size / 1024.0 if Path(name_file).exists() else 0.0

        return {
            "status": "success",
            "scenario_name": scenario_name,
            "combo": combo,
            "combo_tag": combo_tag,
            "scenario_id": chosen.get("scenario_id"),
            "file_hourly": name_file,
            "file_annual": annual_file,
            "elapsed": elapsed,
            "size_kb": size_kb,
            "heat_pump_cop": task.heat_pump_cop if task.use_heat_pump else None,
            "uni_generation_mode": task.uni_generation_mode,
            "runner_type": "ecm_application",
            "report_context": {
                "building_meta": {
                    "name": building_name,
                    "category": building_category,
                    "country": building_country,
                    "weather_source": payload.get("weather_source") or task.weather_source,
                },
                "scenario_meta": {
                    "id": chosen.get("scenario_id"),
                    "label": scenario_label,
                    "description": scenario_description,
                    "combo": list(combo),
                    "elements": chosen.get("elements") or [],
                    "generation_mask": generation_mask,
                },
                "hourly_building": chosen_results.get("hourly_building") or [],
                "primary_energy_uni11300": chosen_uni_results or {},
            },
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": "error",
            "scenario_name": scenario_name,
            "combo": combo,
            "combo_tag": combo_tag,
            "elapsed": elapsed,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "heat_pump_cop": task.heat_pump_cop if task.use_heat_pump else None,
            "uni_generation_mode": task.uni_generation_mode,
            "runner_type": "ecm_application",
        }


def build_scenario_combo(
    *,
    ecm_options: List[str],
    use_heat_pump: bool = False,
    use_pv: bool = False,
    uni_generation_mode: str = "default",
    uni_eta_generation: Optional[float] = None,
) -> List[str]:
    combo = list(extract_envelope_elements(ecm_options))
    if use_heat_pump and "heat_pump" not in combo:
        combo.append("heat_pump")
    if use_pv and "pv" not in combo:
        combo.append("pv")
    if str(uni_generation_mode or "default").strip().lower() == "condensing_boiler":
        combo.append("condensing_boiler")
    elif uni_eta_generation is not None:
        combo.append("eta_generation")
    return sorted(dict.fromkeys(combo))


def _resolve_local_reference_inputs(
    *,
    archetype: bool,
    category: Optional[str],
    country: Optional[str],
    name: Optional[str],
    bui_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if archetype:
        match = next(
            (
                building
                for building in BUILDING_ARCHETYPES
                if building.get("category") == category
                and building.get("country") == country
                and building.get("name") == name
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"No local archetype found for category='{category}', country='{country}', name='{name}'."
            )
        return {
            "building_name": match.get("name") or name or "building",
            "category": match.get("category") or category,
            "country": match.get("country") or country,
            "bui": copy.deepcopy(match.get("bui") or {}),
            "uni11300": copy.deepcopy(
                match.get("uni11300")
                or match.get("systems_archetype")
                or UNI11300_SIMULATION_EXAMPLE
            ),
        }

    if bui_json is None:
        raise ValueError("Custom mode requires bui_json to resolve local PV scenarios.")

    custom_building = bui_json.get("building", {}) if isinstance(bui_json, dict) else {}
    return {
        "building_name": custom_building.get("name") or "custom_building",
        "category": custom_building.get("building_type_class") or category,
        "country": country,
        "bui": copy.deepcopy(bui_json),
        "uni11300": copy.deepcopy(UNI11300_SIMULATION_EXAMPLE),
    }


def _build_uni_config_for_task(
    base_uni_cfg: Optional[Dict[str, Any]],
    *,
    uni_generation_mode: str,
    uni_eta_generation: Optional[float],
) -> Dict[str, Any]:
    uni_cfg = copy.deepcopy(base_uni_cfg or {})
    heating_params = copy.deepcopy(uni_cfg.get("heating_params") or {})

    mode = str(uni_generation_mode or "default").strip().lower()
    if mode == "condensing_boiler":
        heating_params["eta_generation"] = float(
            uni_eta_generation if uni_eta_generation is not None else 1.1
        )
    elif uni_eta_generation is not None:
        heating_params["eta_generation"] = float(uni_eta_generation)

    if heating_params:
        uni_cfg["heating_params"] = heating_params
    return uni_cfg


def _build_integrated_annual_frame(
    hourly_building: pd.DataFrame,
    uni_results: Dict[str, Any],
    pv_hp_results: Dict[str, Any],
) -> pd.DataFrame:
    hourly_df = to_dataframe(hourly_building)
    q_h_kwh = float(
        pd.to_numeric(hourly_df.get("Q_H", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
        .sum()
        * 0.001
    )
    q_c_kwh = float(
        pd.to_numeric(hourly_df.get("Q_C", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
        .sum()
        * 0.001
    )

    uni_summary = uni_results.get("summary") or {}
    pv_summary = (pv_hp_results.get("summary") or {}).get("annual_kwh") or {}
    pv_indicators = (pv_hp_results.get("summary") or {}).get("indicators") or {}

    return pd.DataFrame(
        [
            {
                "Q_H_annual_kWh": q_h_kwh,
                "Q_C_annual_kWh": q_c_kwh,
                "EP_heat_total_kWh": uni_summary.get("EP_heat_total_kWh"),
                "EP_cool_total_kWh": uni_summary.get("EP_cool_total_kWh"),
                "EP_total_kWh": uni_summary.get("EP_total_kWh"),
                "pv_generation_kWh": pv_summary.get("pv_generation"),
                "pv_self_consumption_kWh": pv_summary.get("self_consumption"),
                "pv_grid_import_kWh": pv_summary.get("grid_import"),
                "pv_grid_export_kWh": pv_summary.get("grid_export"),
                "pv_self_consumption_rate": pv_indicators.get("self_consumption_rate"),
                "pv_self_sufficiency_rate": pv_indicators.get("self_sufficiency_rate"),
            }
        ]
    )


def call_iso52016_uni11300_pv(task: ApiTask) -> Dict[str, Any]:
    start = time.time()
    combo = task.combo
    combo_tag = "_".join(sorted(combo)) if combo else "baseline"
    scenario_name = task.scenario_name or combo_tag

    try:
        base_inputs = _resolve_local_reference_inputs(
            archetype=task.archetype,
            category=task.category,
            country=task.country,
            name=task.name,
            bui_json=task.bui_json,
        )
        bui_variant = base_inputs["bui"]
        envelope_elements = extract_envelope_elements(combo)
        if envelope_elements:
            bui_variant = apply_u_values_to_BUI(
                bui_base=bui_variant,
                active_elements=envelope_elements,
                u_wall=task.u_wall,
                u_roof=task.u_roof,
                u_window=task.u_window,
                u_slab=task.u_slab,
            )

        uni_cfg = _build_uni_config_for_task(
            base_inputs.get("uni11300"),
            uni_generation_mode=task.uni_generation_mode,
            uni_eta_generation=task.uni_eta_generation,
        )
        pv_cfg = copy.deepcopy(task.pv_config or DEFAULT_PV_CONFIG)
        payload: Dict[str, Any] = {
            "bui": bui_variant,
            "pv": pv_cfg,
            "uni11300": uni_cfg,
            "return_hourly_building": True,
            "use_heat_pump": bool(task.use_heat_pump),
            "heat_pump_cop": float(task.heat_pump_cop),
        }

        url = f"{BASE_URL}{UNI_PV_ENDPOINT}"
        r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        response_payload = r.json()

        results_payload = response_payload.get("results", {}) if isinstance(response_payload, dict) else {}
        hourly = to_dataframe(results_payload.get("hourly_building"))
        uni_results = results_payload.get("uni11300") or {}
        pv_hp_results = results_payload.get("pv_hp") or {}
        annual = _build_integrated_annual_frame(hourly, uni_results, pv_hp_results)

        building_name = base_inputs.get("building_name") or task.name or "building"
        building_category = base_inputs.get("category") or task.category
        building_country = base_inputs.get("country") or task.country
        scenario_elements = list(envelope_elements)
        if task.use_pv:
            scenario_elements.append("pv")
        name_file = build_name_file(
            building_name=building_name,
            ecm_combo=combo,
            epw_path=None,
            base_dir=task.output_dir,
            scenario_name=scenario_name,
        )
        hourly.to_csv(name_file, index=False)

        annual_file = str(Path(name_file).with_name(Path(name_file).stem + "__annual.csv"))
        annual.to_csv(annual_file, index=False)

        pv_hourly_file = None
        pv_hourly = to_dataframe(pv_hp_results.get("hourly_results"))
        if not pv_hourly.empty:
            pv_hourly_file = str(Path(name_file).with_name(Path(name_file).stem + "__pv_hourly.csv"))
            pv_hourly.to_csv(pv_hourly_file, index=False)

        elapsed = time.time() - start
        size_kb = Path(name_file).stat().st_size / 1024.0 if Path(name_file).exists() else 0.0
        return {
            "status": "success",
            "scenario_name": scenario_name,
            "combo": combo,
            "combo_tag": combo_tag,
            "scenario_id": scenario_name,
            "file_hourly": name_file,
            "file_annual": annual_file,
            "file_pv_hourly": pv_hourly_file,
            "elapsed": elapsed,
            "size_kb": size_kb,
            "heat_pump_cop": task.heat_pump_cop if task.use_heat_pump else None,
            "uni_generation_mode": task.uni_generation_mode,
            "runner_type": "iso52016_uni11300_pv",
            "integrated_results": {
                "uni11300": uni_results,
                "pv_hp": pv_hp_results,
            },
            "report_context": {
                "building_meta": {
                    "name": building_name,
                    "category": building_category,
                    "country": building_country,
                    "weather_source": task.weather_source,
                },
                "scenario_meta": {
                    "id": scenario_name,
                    "label": scenario_name,
                    "description": task.scenario_description or scenario_name,
                    "combo": list(combo),
                    "elements": scenario_elements,
                    "generation_mask": (uni_results.get("generation_mask") if isinstance(uni_results, dict) else None) or {},
                    "pv_config": copy.deepcopy(pv_cfg) if task.use_pv else None,
                },
                "hourly_building": results_payload.get("hourly_building") or [],
                "primary_energy_uni11300": uni_results or {},
            },
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": "error",
            "scenario_name": scenario_name,
            "combo": combo,
            "combo_tag": combo_tag,
            "elapsed": elapsed,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "heat_pump_cop": task.heat_pump_cop if task.use_heat_pump else None,
            "uni_generation_mode": task.uni_generation_mode,
            "runner_type": "iso52016_uni11300_pv",
        }


def _build_generation_suffix(generation_mask: Dict[str, Any]) -> Optional[str]:
    applied_mode = str(generation_mask.get("applied_mode") or "").strip().lower()
    requested_mode = str(generation_mask.get("requested_mode") or "").strip().lower()
    mask_value = generation_mask.get("mask_value")

    if applied_mode == "heat_pump" and mask_value is not None:
        return f"heat pump COP={mask_value}"
    if requested_mode == "condensing_boiler" and mask_value is not None:
        return f"condensing boiler eta_generation={mask_value}"
    if requested_mode not in {"", "default"} and mask_value is not None:
        return f"UNI eta_generation={mask_value}"
    return None


def _normalize_report_scenario_context(result: Dict[str, Any]) -> Dict[str, Any]:
    context = dict(result.get("report_context") or {})
    scenario_meta = dict(context.get("scenario_meta") or {})
    generation_mask = scenario_meta.get("generation_mask") or {}
    generation_suffix = _build_generation_suffix(generation_mask)

    label_parts = [
        str(
            scenario_meta.get("label")
            or result.get("scenario_name")
            or scenario_meta.get("id")
            or ""
        ).strip()
    ]
    if generation_suffix and generation_suffix not in label_parts[0]:
        label_parts.append(generation_suffix)

    scenario_meta["label"] = " + ".join([part for part in label_parts if part]) or "ECM scenario"
    scenario_meta["description"] = str(
        scenario_meta.get("description")
        or scenario_meta.get("label")
        or scenario_meta.get("id")
        or "ECM scenario"
    ).strip()
    context["scenario_meta"] = scenario_meta
    return context


def build_report_html_from_results(
    *,
    results: List[Dict[str, Any]],
    scenario_combo: List[str],
    report_title: str,
    scenario_name: Optional[str] = None,
    scenario_names: Optional[List[str]] = None,
) -> str:
    successful_results = [result for result in results if result.get("status") == "success"]
    baseline_result = next(
        (
            result
            for result in successful_results
            if (result.get("combo_tag") == "baseline") or not result.get("combo")
        ),
        None,
    )
    wanted_combo_tag = "_".join(sorted(scenario_combo)) if scenario_combo else None

    selected_scenario_results: List[Dict[str, Any]] = []
    if scenario_names:
        results_by_name = {
            result.get("scenario_name"): result
            for result in successful_results
            if result.get("scenario_name")
        }
        for selected_name in scenario_names:
            selected = results_by_name.get(selected_name)
            if selected is not None and selected is not baseline_result:
                selected_scenario_results.append(selected)
    else:
        scenario_result = None
        if scenario_name:
            scenario_result = next(
                (
                    result
                    for result in successful_results
                    if result.get("scenario_name") == scenario_name
                ),
                None,
            )
        if scenario_result is None and wanted_combo_tag:
            scenario_result = next(
                (
                    result
                    for result in successful_results
                    if result.get("combo_tag") == wanted_combo_tag
                ),
                None,
            )
        if scenario_result is None:
            scenario_result = next(
                (
                    result
                    for result in successful_results
                    if (result.get("combo_tag") != "baseline") and result.get("combo")
                ),
                None,
            )
        if scenario_result is not None:
            selected_scenario_results = [scenario_result]

    if baseline_result is None:
        raise ValueError("Missing successful baseline result: cannot build HTML report.")
    if not selected_scenario_results:
        raise ValueError("Missing successful ECM scenario result: cannot build HTML report.")

    baseline_context = baseline_result.get("report_context") or {}
    scenario_contexts = [
        _normalize_report_scenario_context(result)
        for result in selected_scenario_results
        if result.get("report_context")
    ]
    if not scenario_contexts:
        raise ValueError("Selected scenarios are missing report context: cannot build HTML report.")

    building_meta = baseline_context.get("building_meta") or scenario_contexts[0].get("building_meta") or {}

    if len(scenario_contexts) == 1:
        scenario_context = scenario_contexts[0]
        scenario_meta = dict(scenario_context.get("scenario_meta") or {})

        return build_ecm_comparison_report_html(
            report_title=report_title,
            building_meta=building_meta,
            scenario_meta=scenario_meta,
            baseline_hourly=to_dataframe(baseline_context.get("hourly_building")),
            scenario_hourly=to_dataframe(scenario_context.get("hourly_building")),
            baseline_uni_results=baseline_context.get("primary_energy_uni11300") or {},
            scenario_uni_results=scenario_context.get("primary_energy_uni11300") or {},
        )

    return build_ecm_multi_scenario_report_html(
        report_title=report_title,
        building_meta=building_meta,
        baseline_hourly=to_dataframe(baseline_context.get("hourly_building")),
        baseline_uni_results=baseline_context.get("primary_energy_uni11300") or {},
        scenario_contexts=scenario_contexts,
    )


def call_ecm_report(
    *,
    ecm_options: List[str],
    u_wall: Optional[float],
    u_roof: Optional[float],
    u_window: Optional[float],
    u_slab: Optional[float],
    weather_source: str,
    epw_path: Optional[Union[str, Path]],
    archetype: bool,
    category: Optional[str],
    country: Optional[str],
    name: Optional[str],
    bui_json: Optional[Dict[str, Any]] = None,
    use_heat_pump: bool = False,
    heat_pump_cop: float = 3.2,
    report_title: Optional[str] = None,
) -> str:
    url = f"{BASE_URL}{REPORT_ENDPOINT}"
    envelope_elements = extract_envelope_elements(ecm_options)

    params: Dict[str, Any] = {
        "archetype": str(archetype).lower(),
        "weather_source": weather_source,
    }
    if u_wall is not None:
        params["u_wall"] = u_wall
    if u_roof is not None:
        params["u_roof"] = u_roof
    if u_window is not None:
        params["u_window"] = u_window
    if u_slab is not None:
        params["u_slab"] = u_slab
    if envelope_elements:
        params["scenario_elements"] = ",".join(envelope_elements)
    if use_heat_pump:
        params["use_heat_pump"] = "true"
        params["heat_pump_cop"] = heat_pump_cop
    if report_title:
        params["report_title"] = report_title

    data: Dict[str, Any] = {}
    files: Dict[str, Any] = {}

    if archetype:
        params.update({"category": category, "country": country, "name": name})
    else:
        if bui_json is None:
            raise ValueError("Custom mode requires bui_json")
        data["bui_json"] = json.dumps(bui_json)

    if weather_source == "epw":
        if not epw_path:
            raise ValueError("weather_source='epw' requires epw_path")
        epw_p = Path(epw_path)
        files["epw_file"] = (epw_p.name, epw_p.read_bytes(), "application/octet-stream")

    r = requests.post(url, params=params, data=data, files=files if files else None, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.text


# =============================================================================
# MAIN ORCHESTRATION (SEQUENTIAL)
# =============================================================================

def run_predefined_renovation_scenarios(
    *,
    scenarios: List[RenovationScenario],
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
    ensure_dir(output_dir)
    ensure_dir(BUILDING_EXAMPLES_DIR)

    tasks: List[tuple[str, ApiTask]] = []
    if include_baseline:
        tasks.append(
            (
                "ecm_application",
                ApiTask(
                    scenario_name="baseline",
                    combo=[],
                    u_wall=None,
                    u_roof=None,
                    u_window=None,
                    u_slab=None,
                    weather_source=weather_source,
                    epw_path=str(epw_path) if epw_path else None,
                    output_dir=str(output_dir),
                    archetype=archetype,
                    category=category,
                    country=country,
                    name=name,
                    bui_json=bui_json,
                    scenario_elements=None,
                    baseline_only=True,
                    include_baseline=False,
                    use_heat_pump=False,
                    heat_pump_cop=HEAT_PUMP_COP_DEFAULT,
                    uni_generation_mode="default",
                    uni_eta_generation=None,
                    scenario_description="Baseline reference scenario",
                ),
            )
        )

    for scenario in scenarios:
        combo = build_scenario_combo(
            ecm_options=scenario.ecm_options,
            use_heat_pump=scenario.use_heat_pump,
            use_pv=scenario.use_pv,
            uni_generation_mode=scenario.uni_generation_mode,
            uni_eta_generation=scenario.uni_eta_generation,
        )
        envelope_elements = extract_envelope_elements(scenario.ecm_options)
        generation_only = (
            not envelope_elements
            and (
                scenario.use_heat_pump
                or scenario.use_pv
                or str(scenario.uni_generation_mode or "default").strip().lower() != "default"
                or scenario.uni_eta_generation is not None
            )
        )
        task = ApiTask(
            scenario_name=scenario.name,
            combo=combo,
            u_wall=U_WALL if "wall" in scenario.ecm_options else None,
            u_roof=U_ROOF if "roof" in scenario.ecm_options else None,
            u_window=U_WINDOW if "window" in scenario.ecm_options else None,
            u_slab=U_SLAB if "slab" in scenario.ecm_options else None,
            weather_source=weather_source,
            epw_path=str(epw_path) if epw_path else None,
            output_dir=str(output_dir),
            archetype=archetype,
            category=category,
            country=country,
            name=name,
            bui_json=bui_json,
            scenario_elements=",".join(envelope_elements) if envelope_elements else None,
            baseline_only=False,
            include_baseline=generation_only and not scenario.use_pv,
            use_heat_pump=scenario.use_heat_pump,
            heat_pump_cop=scenario.heat_pump_cop,
            uni_generation_mode=scenario.uni_generation_mode,
            uni_eta_generation=scenario.uni_eta_generation,
            scenario_description=scenario.description,
        )
        task.pv_config = copy.deepcopy(scenario.pv_config) if scenario.pv_config else None
        runner_type = "iso52016_uni11300_pv" if scenario.use_pv else "ecm_application"
        tasks.append((runner_type, task))

    start = time.time()
    results: List[Dict[str, Any]] = []
    iterator = tasks
    if tqdm:
        iterator = tqdm(tasks, desc="Renovation scenarios", total=len(tasks))

    for runner_type, task in iterator:
        if runner_type == "iso52016_uni11300_pv":
            results.append(call_iso52016_uni11300_pv(task))
        else:
            results.append(call_ecm_application(task))

    total_time = time.time() - start
    ok = [r for r in results if r["status"] == "success"]
    ko = [r for r in results if r["status"] == "error"]

    return {
        "total": len(results),
        "successful": len(ok),
        "failed": len(ko),
        "total_time": total_time,
        "results": results,
        "scenario_combo": [],
        "scenario_names": [scenario.name for scenario in scenarios],
    }


def run_ecm_api_sequential(
    ecm_options: List[str],
    u_wall: Optional[float],
    u_roof: Optional[float],
    u_window: Optional[float],
    u_slab: Optional[float],
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
    allowed = {"wall", "roof", "window", "slab", "heat_pump"}
    bad = [o for o in ecm_options if o not in allowed]
    if bad:
        raise ValueError(f"Invalid ECM option(s): {bad}. Allowed: {sorted(list(allowed))}.")

    # validations
    if "wall" in ecm_options and u_wall is None:
        raise ValueError("For ECM 'wall' you must provide u_wall.")
    if "roof" in ecm_options and u_roof is None:
        raise ValueError("For ECM 'roof' you must provide u_roof.")
    if "window" in ecm_options and u_window is None:
        raise ValueError("For ECM 'window' you must provide u_window.")
    if "slab" in ecm_options and u_slab is None:
        raise ValueError("For ECM 'slab' you must provide u_slab.")

    ensure_dir(output_dir)
    ensure_dir(BUILDING_EXAMPLES_DIR)

    envelope_options = extract_envelope_elements(ecm_options)
    scenario_use_heat_pump = ("heat_pump" in ecm_options) or USE_HEAT_PUMP
    scenario_combo = list(envelope_options)
    if scenario_use_heat_pump:
        scenario_combo.append("heat_pump")

    if not include_baseline and not scenario_combo:
        raise ValueError("Nothing to run: enable baseline and/or provide at least one ECM option.")

    tasks: List[ApiTask] = []
    if include_baseline:
        tasks.append(
            ApiTask(
                scenario_name="baseline",
                combo=[],
                u_wall=u_wall if ("wall" in ecm_options) else None,
                u_roof=u_roof if ("roof" in ecm_options) else None,
                u_window=u_window if ("window" in ecm_options) else None,
                u_slab=u_slab if ("slab" in ecm_options) else None,
                weather_source=weather_source,
                epw_path=str(epw_path) if epw_path else None,
                output_dir=str(output_dir),
                archetype=archetype,
                category=category,
                country=country,
                name=name,
                bui_json=bui_json,
                scenario_elements=None,
                baseline_only=True,
                include_baseline=False,
                use_heat_pump=False,
                heat_pump_cop=HEAT_PUMP_COP_DEFAULT,
                uni_generation_mode="default",
                uni_eta_generation=None,
            )
        )

    if scenario_combo:
        if not archetype and bui_json is not None and envelope_options:
            _ = apply_u_values_to_BUI(
                bui_base=bui_json,
                active_elements=envelope_options,
                u_wall=u_wall,
                u_roof=u_roof,
                u_window=u_window,
                u_slab=u_slab,
            )

        combo_tag = "_".join(sorted(scenario_combo))
        cop = HEAT_PUMP_COP_BY_COMBO.get(combo_tag, HEAT_PUMP_COP_DEFAULT)

        tasks.append(
            ApiTask(
                scenario_name="_".join(sorted(scenario_combo)) if scenario_combo else "scenario",
                combo=scenario_combo,
                u_wall=u_wall if ("wall" in ecm_options) else None,
                u_roof=u_roof if ("roof" in ecm_options) else None,
                u_window=u_window if ("window" in ecm_options) else None,
                u_slab=u_slab if ("slab" in ecm_options) else None,
                weather_source=weather_source,
                epw_path=str(epw_path) if epw_path else None,
                output_dir=str(output_dir),
                archetype=archetype,
                category=category,
                country=country,
                name=name,
                bui_json=bui_json,
                scenario_elements=",".join(envelope_options) if envelope_options else None,
                baseline_only=False,
                include_baseline=False,
                use_heat_pump=scenario_use_heat_pump,
                heat_pump_cop=cop,
                uni_generation_mode="default",
                uni_eta_generation=None,
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
        "scenario_combo": scenario_combo,
    }


def build_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "scenario_name": r.get("scenario_name"),
                "runner_type": r.get("runner_type"),
                "combo": r.get("combo_tag") or ("_".join(sorted(r.get("combo", []))) if r.get("combo") else "baseline"),
                "status": r.get("status"),
                "scenario_id": r.get("scenario_id"),
                "elapsed_s": round(float(r.get("elapsed", 0.0)), 3) if r.get("elapsed") is not None else None,
                "size_kb": round(float(r.get("size_kb", 0.0)), 2) if r.get("size_kb") is not None else None,
                "heat_pump_cop": r.get("heat_pump_cop"),
                "uni_generation_mode": r.get("uni_generation_mode"),
                "file_hourly": r.get("file_hourly"),
                "file_annual": r.get("file_annual"),
                "file_pv_hourly": r.get("file_pv_hourly"),
                "error": r.get("error"),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# RUN
# =============================================================================

def main() -> Dict[str, Any]:
    if USE_RENOVATION_SCENARIO_LIBRARY:
        stats = run_predefined_renovation_scenarios(
            scenarios=RENOVATION_SCENARIOS,
            weather_source=WEATHER_SOURCE,
            epw_path=EPW_PATH if WEATHER_SOURCE == "epw" else None,
            output_dir=RESULTS_DIR,
            include_baseline=INCLUDE_BASELINE,
            archetype=ARCHETYPE,
            category=CATEGORY,
            country=COUNTRY,
            name=ARCHETYPE_NAME,
        )
    else:
        stats = run_ecm_api_sequential(
            ecm_options=ECM_OPTIONS,
            u_wall=U_WALL,
            u_roof=U_ROOF,
            u_window=U_WINDOW,
            u_slab=U_SLAB,
            weather_source=WEATHER_SOURCE,
            epw_path=EPW_PATH if WEATHER_SOURCE == "epw" else None,
            output_dir=RESULTS_DIR,
            include_baseline=INCLUDE_BASELINE,
            archetype=ARCHETYPE,
            category=CATEGORY,
            country=COUNTRY,
            name=ARCHETYPE_NAME,
        )

    report_path = None
    report_error = None
    scenario_use_heat_pump = ("heat_pump" in ECM_OPTIONS) or USE_HEAT_PUMP
    report_scenario_names: List[str] = []
    if USE_RENOVATION_SCENARIO_LIBRARY:
        if REPORT_SCENARIO_NAME:
            report_scenario_names = [REPORT_SCENARIO_NAME]
        else:
            report_scenario_names = list(stats.get("scenario_names", []))
    should_generate_report = (
        GENERATE_REPORT
        and (
            (USE_RENOVATION_SCENARIO_LIBRARY and bool(report_scenario_names))
            or bool(stats.get("scenario_combo"))
            or scenario_use_heat_pump
        )
    )
    if should_generate_report:
        try:
            report_html = build_report_html_from_results(
                results=stats.get("results", []),
                scenario_combo=stats.get("scenario_combo", []),
                report_title=REPORT_TITLE,
                scenario_name=report_scenario_names[0] if len(report_scenario_names) == 1 else None,
                scenario_names=report_scenario_names if len(report_scenario_names) > 1 else None,
            )
            report_tag = None
            if USE_RENOVATION_SCENARIO_LIBRARY and report_scenario_names:
                report_tag = "_".join(report_scenario_names)
            report_path = build_report_file(
                building_name=ARCHETYPE_NAME if ARCHETYPE else "custom_building",
                ecm_combo=stats.get("scenario_combo", []),
                epw_path=EPW_PATH if WEATHER_SOURCE == "epw" else None,
                base_dir=RESULTS_DIR,
                scenario_name=report_tag if USE_RENOVATION_SCENARIO_LIBRARY else None,
            )
            Path(report_path).write_text(report_html, encoding="utf-8")
        except ValueError as exc:
            report_error = str(exc)

    print("\n" + "=" * 80)
    print("COMPLETED (ECM API)")
    print("=" * 80)
    print(f"Successes: {stats['successful']}/{stats['total']}")
    print(f"Failed: {stats['failed']}/{stats['total']}")
    print(f"Time: {stats['total_time']:.1f}s")
    if USE_RENOVATION_SCENARIO_LIBRARY:
        print(f"Scenario library: {', '.join(stats.get('scenario_names', []))}")

    summary_df = build_summary_table(stats["results"])
    if not summary_df.empty:
        print("\nSUMMARY TABLE (per simulation)")
        print(summary_df.to_string(index=False))
    if report_path:
        print(f"\nHTML REPORT: {report_path}")
    elif report_error:
        print(f"\nHTML REPORT SKIPPED: {report_error}")

    if stats["failed"]:
        print("\n[ERRORS]")
        for r in stats["results"]:
            if r["status"] == "error":
                print(f"- combo={r.get('combo')}: {r.get('error')}")

    return {"stats": stats, "report_path": report_path, "report_error": report_error}


if __name__ == "__main__":
    main()
