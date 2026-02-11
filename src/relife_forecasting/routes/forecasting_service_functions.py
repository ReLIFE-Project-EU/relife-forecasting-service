# forecasting_services.py
from __future__ import annotations

import copy
import itertools
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pybuildingenergy as pybui
from fastapi import HTTPException
from pydantic import BaseModel, Field

from relife_forecasting.utils.retry import retry_on_transient_error


# =============================================================================
# JSON / Data utilities
# =============================================================================

def to_jsonable(obj: Any) -> Any:
    """
    Convert an arbitrary Python object into a JSON-serializable structure.

    This is a permissive converter intended for returning complex objects
    (numpy/pandas outputs) in API responses.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()

    if isinstance(obj, pd.DataFrame):
        return {col: to_jsonable(obj[col].values) for col in obj.columns}

    try:
        return obj.tolist()
    except AttributeError:
        return str(obj)


def clean_and_jsonable(obj: Any) -> Any:
    """
    Convert complex objects (numpy/pandas/etc.) into JSON-safe structures.

    Compared to `to_jsonable`, this function makes an explicit effort to:
      - keep numeric arrays as floats
      - recursively sanitize object arrays
      - sanitize DataFrames into column-wise dicts
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return {k: clean_and_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [clean_and_jsonable(x) for x in obj]

    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.number):
            return obj.astype(float).tolist()
        return [clean_and_jsonable(x) for x in obj.tolist()]

    if isinstance(obj, np.generic):
        if np.issubdtype(type(obj), np.number):
            return float(obj)
        return obj.item()

    if isinstance(obj, (pd.Series, pd.Index)):
        return [clean_and_jsonable(x) for x in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return {col: clean_and_jsonable(obj[col].values) for col in obj.columns}

    try:
        return obj.tolist()
    except AttributeError:
        return str(obj)


def dict_to_df_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a pandas DataFrame into a list of row dictionaries (records),
    which is often more convenient for JSON API responses.
    """
    return df.to_dict(orient="records")


def dataframe_to_records_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame into JSON-safe records.

    Behavior:
      - If index is DatetimeIndex, it is preserved into a `timestamp` column.
      - Replaces inf/-inf with NaN, then NaN with None.
      - Converts datetime columns to ISO-like strings.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "timestamp"})
    else:
        df = df.reset_index(drop=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

    for col in df.columns:
        if df[col].dtype == "datetime64[ns]":
            df[col] = df[col].astype(str)

    return df.to_dict(orient="records")


# =============================================================================
# JSON -> internal pybuildingenergy structures
# =============================================================================

def json_to_internal_bui(bui_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a JSON BUI representation into the internal format expected by pybuildingenergy.

    Note:
      - Some pybuildingenergy fields are expected to be numpy arrays (dtype=object).
      - This function converts specific adjacent_zones list fields into numpy arrays.
    """
    bui = copy.deepcopy(bui_json)

    for zone in bui.get("adjacent_zones", []):
        for key in (
            "area_facade_elements",
            "typology_elements",
            "transmittance_U_elements",
            "orientation_elements",
        ):
            if key in zone and isinstance(zone[key], list):
                zone[key] = np.array(zone[key], dtype=object)

    return bui


def json_to_internal_system(system_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a JSON system representation into the internal format expected by pybuildingenergy.

    Note:
      All three DataFrame fields may arrive from the API in one of two JSON shapes:
        - list[dict] (records format, e.g. from explicit serialisation)
        - dict[str, list] (column-oriented format produced by to_jsonable())
      Both shapes are handled by passing the value directly to pd.DataFrame(), which
      understands both.  The index is then assigned explicitly because to_jsonable()
      discards the row labels that pybuildingenergy expects.

      - `gen_outdoor_temp_data`: 1-row DataFrame; index = ["Generator curve"]
      - `heat_emission_data`: 5-row DataFrame; index = [
          "Max flow temperature HZ1", "Max Δθ flow / return HZ1",
          "Desired return temperature HZ1", "Desired load factor with ON-OFF for HZ1",
          "Minimum flow temperature for HZ1"]
      - `outdoor_temp_data`: 4-row DataFrame; index = [
          "Minimum outdoor temperature", "Maximum outdoor temperature",
          "Maximum flow temperature", "Minimum flow temperature"]
    """
    system = copy.deepcopy(system_json)

    if "gen_outdoor_temp_data" in system and not isinstance(system["gen_outdoor_temp_data"], pd.DataFrame):
        df = pd.DataFrame(system["gen_outdoor_temp_data"])
        df.index = ["Generator curve"] * len(df)
        system["gen_outdoor_temp_data"] = df

    if "heat_emission_data" in system and not isinstance(system["heat_emission_data"], pd.DataFrame):
        df = pd.DataFrame(system["heat_emission_data"])
        df.index = [
            "Max flow temperature HZ1",
            "Max Δθ flow / return HZ1",
            "Desired return temperature HZ1",
            "Desired load factor with ON-OFF for HZ1",
            "Minimum flow temperature for HZ1",
        ]
        system["heat_emission_data"] = df

    if "outdoor_temp_data" in system and not isinstance(system["outdoor_temp_data"], pd.DataFrame):
        df = pd.DataFrame(system["outdoor_temp_data"])
        df.index = [
            "Minimum outdoor temperature",
            "Maximum outdoor temperature",
            "Maximum flow temperature",
            "Minimum flow temperature",
        ]
        system["outdoor_temp_data"] = df

    return system


# =============================================================================
# Validation helpers
# =============================================================================

def validate_bui_and_system(bui: Dict[str, Any], system: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate both BUI (building input) and HVAC system configuration.

    Steps:
      1) If available, run `pybui.check_heating_system_inputs(system)` to normalize/validate HVAC inputs.
      2) Run `pybui.sanitize_and_validate_BUI(bui, fix=True)` to auto-fix some issues.
      3) Run `pybui.sanitize_and_validate_BUI(bui_fixed, fix=False)` to validate without fixes.
      4) If any ERROR-level issues are found, raise HTTP 422.

    Returns:
      A dict containing fixed/checked objects and validation reports/messages.
    """
    system_messages: List[str] = []
    system_checked = system

    try:
        res_sys = pybui.check_heating_system_inputs(system)
        system_checked = res_sys["config"]
        system_messages.extend(res_sys.get("messages", []))
    except AttributeError:
        system_messages.append(
            "Function 'check_heating_system_inputs' is not available in the installed "
            "pybuildingenergy version: HVAC inputs are not validated automatically."
        )

    bui_fixed, report_fixed = pybui.sanitize_and_validate_BUI(bui, fix=True)
    bui_checked, issues = pybui.sanitize_and_validate_BUI(bui_fixed, fix=False)

    errors = [e for e in issues if e.get("level") == "ERROR"]
    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "msg": "Errors in BUI model.",
                "errors": errors,
                "system_messages": system_messages,
            },
        )

    return {
        "bui_fixed": bui_fixed,
        "bui_checked": bui_checked,
        "system_checked": system_checked,
        "bui_report_fixed": report_fixed,
        "bui_issues": issues,
        "system_messages": system_messages,
    }


# =============================================================================
# Simulation worker (multiprocessing-safe)
# =============================================================================

def simulate_building_worker(building_name: str, bui: dict, system: dict) -> Dict[str, Any]:
    """
    Multiprocessing worker: simulate a single building (ISO 52016 + ISO 15316).

    Returns:
      {
        "name": str,
        "status": "ok"|"error",
        "summary": {...},
        "hourly": <jsonable>,
        "annual": <jsonable>,
        "system": <jsonable>
      }
    """
    try:

        @retry_on_transient_error()
        def _run_worker_pvgis():
            return pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                bui,
                weather_source="pvgis",
            )

        hourly_sim, annual_results_df = _run_worker_pvgis()

        calc = pybui.HeatingSystemCalculator(system)
        calc.load_csv_data(hourly_sim)
        df_system = calc.run_timeseries()

        heating_kWh = hourly_sim.loc[hourly_sim["Q_HC"] > 0, "Q_HC"].sum() / 1000
        cooling_kWh = -hourly_sim.loc[hourly_sim["Q_HC"] < 0, "Q_HC"].sum() / 1000
        area = bui.get("building", {}).get("net_floor_area")

        summary = {
            "name": building_name,
            "heating_kWh": heating_kWh,
            "cooling_kWh": cooling_kWh,
            "area": area,
            "heating_kWh_m2": heating_kWh / area if area else None,
            "cooling_kWh_m2": cooling_kWh / area if area else None,
        }

        return {
            "name": building_name,
            "status": "ok",
            "summary": summary,
            "hourly": clean_and_jsonable(hourly_sim),
            "annual": clean_and_jsonable(annual_results_df),
            "system": clean_and_jsonable(df_system),
        }

    except Exception as e:
        return {"name": building_name, "status": "error", "error": str(e)}


# =============================================================================
# ECM helpers (U-values manipulation)
# =============================================================================

def classify_surface(surface: Dict[str, Any]) -> Optional[str]:
    """
    Classify a building surface based on simplified geometric/type rules.

    Returns:
      - "roof"   for opaque surfaces with tilt=0 and azimuth=0
      - "slab"   for opaque ground slabs (tilt=0, azimuth=0, typically sky_view_factor=0)
      - "wall"   for opaque surfaces with tilt=90
      - "window" for transparent surfaces with tilt=90
      - None     for non-matching surfaces
    """
    s_type = surface.get("type")
    orientation = surface.get("orientation", {}) or {}
    tilt = orientation.get("tilt")
    azimuth = orientation.get("azimuth")
    name = str(surface.get("name") or "").lower()
    sky_view = surface.get("sky_view_factor")

    if s_type == "opaque" and tilt == 0 and azimuth == 0:
        if ("slab" in name) or ("ground" in name) or (sky_view == 0):
            return "slab"
        return "roof"

    if s_type == "opaque" and tilt == 90:
        return "wall"

    if s_type == "transparent" and tilt == 90:
        return "window"

    return None


def apply_u_values_to_bui(
    bui: Dict[str, Any],
    use_roof: bool,
    use_wall: bool,
    use_window: bool,
    u_roof: Optional[float],
    u_wall: Optional[float],
    u_window: Optional[float],
    use_slab: bool = False,
    u_slab: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return a deep-copied BUI with updated U-values for the selected surface categories.
    """
    bui_copy = copy.deepcopy(bui)

    for surface in bui_copy.get("building_surface", []):
        kind = classify_surface(surface)

        if kind == "roof" and use_roof and u_roof is not None:
            surface["u_value"] = float(u_roof)
        elif kind == "wall" and use_wall and u_wall is not None:
            surface["u_value"] = float(u_wall)
        elif kind == "window" and use_window and u_window is not None:
            surface["u_value"] = float(u_window)
        elif kind == "slab" and use_slab and u_slab is not None:
            surface["u_value"] = float(u_slab)

    return bui_copy


def build_uvalue_scenarios(
    u_roof: Optional[float],
    u_wall: Optional[float],
    u_window: Optional[float],
    u_slab: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Generate all non-empty combinations of requested U-value interventions.

    Returns:
      A list of scenario specs, each containing:
        - id
        - use_roof/use_wall/use_window/use_slab
        - label
    """
    u_map = {"roof": u_roof, "wall": u_wall, "window": u_window, "slab": u_slab}
    active_types = [k for k, v in u_map.items() if v is not None]

    if not active_types:
        return []

    scenarios_spec: List[Dict[str, Any]] = []
    for r in range(1, len(active_types) + 1):
        for subset in itertools.combinations(active_types, r):
            subset_set = set(subset)
            use_roof = "roof" in subset_set
            use_wall = "wall" in subset_set
            use_window = "window" in subset_set
            use_slab = "slab" in subset_set

            label_parts: List[str] = []
            if use_roof:
                label_parts.append(f"roof U={u_roof}")
            if use_wall:
                label_parts.append(f"walls U={u_wall}")
            if use_window:
                label_parts.append(f"windows U={u_window}")
            if use_slab:
                label_parts.append(f"slab U={u_slab}")

            scenarios_spec.append(
                {
                    "id": "+".join(sorted(subset_set)),
                    "use_roof": use_roof,
                    "use_wall": use_wall,
                    "use_window": use_window,
                    "use_slab": use_slab,
                    "label": ", ".join(label_parts),
                }
            )

    return scenarios_spec


def apply_heat_pump_to_system(system: Dict[str, Any], cop: float = 3.2) -> Dict[str, Any]:
    """
    Return a deep-copied system dict updated to represent a heat pump generator.

    Notes:
      - Adds generator metadata and a COP value.
      - Sets a parametric efficiency model with eta_max / eta_no_cond expressed in %.
    """
    sys_copy = copy.deepcopy(system)
    sys_copy["generator_type"] = "heat_pump"
    sys_copy["heat_pump_cop"] = float(cop)
    sys_copy["efficiency_model"] = "parametric"
    sys_copy["eta_max"] = float(cop) * 100.0
    sys_copy["eta_no_cond"] = float(cop) * 100.0
    return sys_copy


# =============================================================================
# CO2 / emissions models & business logic
# =============================================================================

class EnergySource(str, Enum):
    """Supported energy sources for emissions accounting."""
    GRID_ELECTRICITY = "grid_electricity"
    NATURAL_GAS = "natural_gas"
    LPG = "lpg"
    DIESEL = "diesel"
    BIOMASS = "biomass"
    DISTRICT_HEATING = "district_heating"
    SOLAR_PV = "solar_pv"
    WIND = "wind"
    HEAT_PUMP_ELECTRIC = "heat_pump_electric"


EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "IT": {
        "grid_electricity": 0.280,
        "natural_gas": 0.202,
        "lpg": 0.234,
        "diesel": 0.267,
        "biomass": 0.030,
        "district_heating": 0.180,
        "solar_pv": 0.040,
        "wind": 0.012,
        "heat_pump_electric": 0.070,
    },
    "EU": {
        "grid_electricity": 0.255,
        "natural_gas": 0.202,
        "lpg": 0.234,
        "diesel": 0.267,
        "biomass": 0.030,
        "district_heating": 0.150,
        "solar_pv": 0.040,
        "wind": 0.012,
        "heat_pump_electric": 0.064,
    },
    "DE": {
        "grid_electricity": 0.420,
        "natural_gas": 0.202,
        "lpg": 0.234,
        "diesel": 0.267,
        "biomass": 0.030,
        "district_heating": 0.200,
        "solar_pv": 0.040,
        "wind": 0.012,
        "heat_pump_electric": 0.105,
    },
}


class ScenarioInput(BaseModel):
    """Input payload for a single scenario emissions calculation."""
    name: str = Field(..., description="Scenario name")
    energy_source: EnergySource = Field(..., description="Energy source")
    annual_consumption_kwh: float = Field(..., gt=0, description="Annual consumption [kWh]")
    country: str = Field("IT", description="Country code (IT, EU, DE)")


class MultiScenarioInput(BaseModel):
    """Input payload for comparing multiple scenarios (first scenario is baseline)."""
    scenarios: List[ScenarioInput] = Field(..., min_length=1, max_length=10)


class EmissionResult(BaseModel):
    """Response model for a single emissions calculation."""
    name: str
    energy_source: str
    annual_consumption_kwh: float
    emission_factor_kg_per_kwh: float
    annual_emissions_kg_co2eq: float
    annual_emissions_ton_co2eq: float
    equivalent_trees: int
    equivalent_km_car: int


class SavingResult(BaseModel):
    """Response model for savings vs. baseline."""
    scenario_name: str
    absolute_kg_co2eq: float
    absolute_ton_co2eq: float
    percentage: float


class ComparisonResult(BaseModel):
    """Response model for a multi-scenario comparison."""
    baseline: EmissionResult
    scenarios: List[EmissionResult]
    best_scenario: str
    savings: List[SavingResult]


class InterventionInput(BaseModel):
    """Input payload to evaluate a retrofit intervention impact on emissions."""
    current_consumption_kwh: float = Field(..., gt=0, description="Current consumption [kWh/year]")
    current_source: EnergySource = Field(..., description="Current energy source")
    energy_reduction_percentage: float = Field(0, ge=0, le=100, description="Consumption reduction [%]")
    new_source: Optional[EnergySource] = Field(None, description="New energy source (optional)")
    new_consumption_kwh: Optional[float] = Field(None, description="New annual consumption (optional)")
    country: str = Field("IT", description="Country code for emission factors (IT, EU, DE)")


def calculate_emissions(
    energy_source: Union[EnergySource, str],
    annual_consumption_kwh: float,
    country: str = "IT",
) -> Dict[str, float]:
    """
    Compute CO2e emissions and practical equivalent metrics.

    Returns:
      {
        "emission_factor": float,
        "annual_emissions_kg": float,
        "annual_emissions_ton": float,
        "equivalent_trees": int,
        "equivalent_km_car": int
      }
    """
    if country not in EMISSION_FACTORS:
        country = "IT"

    source_key = energy_source.value if isinstance(energy_source, EnergySource) else str(energy_source)

    emission_factor = EMISSION_FACTORS[country].get(
        source_key,
        EMISSION_FACTORS[country]["grid_electricity"],
    )

    annual_emissions_kg = annual_consumption_kwh * emission_factor
    annual_emissions_ton = annual_emissions_kg / 1000.0

    # Practical equivalents (heuristics)
    equivalent_trees = int(annual_emissions_kg / 21.0)   # ~21 kg CO2 per tree per year
    equivalent_km_car = int(annual_emissions_kg / 0.120) # ~120 g CO2 per km

    return {
        "emission_factor": emission_factor,
        "annual_emissions_kg": annual_emissions_kg,
        "annual_emissions_ton": annual_emissions_ton,
        "equivalent_trees": equivalent_trees,
        "equivalent_km_car": equivalent_km_car,
    }


def calculate_savings(baseline_emissions: float, scenario_emissions: float) -> Dict[str, float]:
    """
    Compute absolute and relative savings between a baseline and an alternative scenario.
    """
    absolute_saving = baseline_emissions - scenario_emissions
    percentage_saving = (absolute_saving / baseline_emissions * 100.0) if baseline_emissions > 0 else 0.0

    return {
        "absolute_kg_co2eq": round(absolute_saving, 2),
        "absolute_ton_co2eq": round(absolute_saving / 1000.0, 3),
        "percentage": round(percentage_saving, 1),
    }
