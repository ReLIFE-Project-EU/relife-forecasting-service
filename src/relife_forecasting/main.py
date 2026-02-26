# main.py
from __future__ import annotations

from importlib.metadata import version
import copy
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import pybuildingenergy as pybui
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from relife_forecasting.config.logging import configure_logging
from relife_forecasting.routes import health
try: 
    from relife_forecasting.utils.retry import retry_on_transient_error
except Exception:
    from utils.retry import retry_on_transient_error

# Prefer your package imports; keep a small fallback to reduce friction during refactors.
try:
    from relife_forecasting.building_examples import BUILDING_ARCHETYPES, UNI11300_SIMULATION_EXAMPLE
except Exception:
    from building_examples import BUILDING_ARCHETYPES, UNI11300_SIMULATION_EXAMPLE  # type: ignore

try:
    from relife_forecasting.routes.EPC_Greece_converter import U_VALUES_BY_CLASS, _norm_surface_name
except Exception:
    from routes.EPC_Greece_converter import U_VALUES_BY_CLASS, _norm_surface_name  # type: ignore

try:
    from relife_forecasting.routes.forecasting_service_functions import *
except Exception:
    from routes.forecasting_service_functions import *

try:
    from relife_forecasting.routes.uni11300_primary_energy import (
        HeatingSystemParams,
        CoolingSystemParams,
        compute_primary_energy_from_hourly_ideal,
        build_uni11300_input_example,
    )
except Exception:
    from routes.uni11300_primary_energy import (
        HeatingSystemParams,
        CoolingSystemParams,
        compute_primary_energy_from_hourly_ideal,
        build_uni11300_input_example,
    )


from relife_forecasting.routes import health



# -----------------------------------------------------------------------------
# Version + logging
# -----------------------------------------------------------------------------

package_name = __name__.split(".")[0]
package_dist_name = package_name.replace("_", "-")

try:
    __version__ = version(package_dist_name)
except Exception:
    __version__ = "development"

configure_logging()


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="ReLIFE Forecasting Service",
    description="""
This service evaluates the energy and thermal comfort performance of individual buildings and entire building stocks, 
providing detailed projections for the years **2030** and **2050**.

**Main objectives:**
- Identify cost-effective renovation and retrofit strategies
- Improve overall building performance and increase economic value
- Significantly reduce energy demand and CO₂ emissions
- Maximize the integration and use of renewable energy sources (RES)
- Deliver environmental and health co-benefits, including:
  - Reduction of greenhouse gas (GHG) emissions
  - Decrease in heat- and cold-related illnesses and discomfort

The analysis is performed using advanced **dynamic energy simulation tools** (EnergyPlus-based) 
enhanced and calibrated within the ReLIFE project framework.
""",
    version=__version__,
)

app.include_router(health.router)


# -----------------------------------------------------------------------------
# In-memory storage (demo only)
# -----------------------------------------------------------------------------

PROJECTS: Dict[str, Any] = {}


# =============================================================================
# Building & archetypes endpoints
# =============================================================================

@app.post("/building", tags=["Building and systems"])
def get_building_config(
    archetype: bool = Query(
        True,
        description="If True, use BUI/HVAC from internal archetypes; if False, use custom values from the request body.",
    ),
    category: Optional[str] = Query(
        None,
        description="Building category (Single Family House, Multi family House, office). Required when archetype=True.",
    ),
    country: Optional[str] = Query(
        None,
        description="Country (Italy, Greece, etc.). Required when archetype=True.",
    ),
    name: Optional[str] = Query(
        None,
        description="Specific archetype name. Required when archetype=True.",
    ),
    payload: Optional[Dict[str, Any]] = Body(
        None,
        description="JSON body with 'bui' and 'system' when archetype=False.",
    ),
):
    """
    Return example BUI/System either from internal archetypes or from a custom payload.

    - If archetype=True: fetch BUI/system from BUILDING_ARCHETYPES by category/country/name.
    - If archetype=False: echo back the custom BUI/system passed in the request body.
    """
    if archetype:
        if not category or not country or not name:
            raise HTTPException(
                status_code=400,
                detail="With archetype=true you must provide 'category', 'country' and 'name' as query parameters.",
            )

        match = next(
            (b for b in BUILDING_ARCHETYPES if b.get("category") == category and b.get("country") == country and b.get("name") == name),
            None,
        )
        if match is None:
            raise HTTPException(
                status_code=404,
                detail=f"No archetype found for category='{category}', country='{country}', name='{name}'.",
            )

        return {
            "source": "archetype",
            "name": match["name"],
            "category": match["category"],
            "country": match["country"],
            "bui": to_jsonable(match["bui"]),
            "system": to_jsonable(match["system"]),
            "uni11300_input_example": to_jsonable(
                match.get("uni11300")
                or match.get("systems_archetype")
                or UNI11300_SIMULATION_EXAMPLE
            ),
        }

    if payload is None:
        raise HTTPException(
            status_code=400,
            detail="With archetype=false you must send a JSON body containing 'bui' and 'system'.",
        )

    bui_json = payload.get("bui")
    system_json = payload.get("system")
    if bui_json is None or system_json is None:
        raise HTTPException(status_code=400, detail="Incomplete JSON body: both 'bui' and 'system' are required.")

    return {
        "source": "custom",
        "name": name,
        "category": category,
        "country": country,
        "bui": bui_json,
        "system": system_json,
    }


@app.get("/building/available", tags=["Building and systems"])
def list_available_archetypes():
    """
    List available archetypes (metadata only, without full BUI/HVAC content).
    """
    return [{"name": b.get("name"), "category": b.get("category"), "country": b.get("country")} for b in BUILDING_ARCHETYPES]


# =============================================================================
# Validation endpoint
# =============================================================================

@app.post("/validate", tags=["Validation"])
def validate_model(
    archetype: bool = Query(True, description="If True, validate an archetype; if False, validate a custom model from the body."),
    category: Optional[str] = Query(None, description="Building category. Required when archetype=True."),
    country: Optional[str] = Query(None, description="Country. Required when archetype=True."),
    name: Optional[str] = Query(None, description="Archetype name. Required when archetype=True."),
    payload: Optional[Dict[str, Any]] = Body(None, description="JSON body with 'bui' and 'system' when archetype=False."),
):
    """
    Validate BUI and HVAC system for either:
      - an archetype (when archetype=True), or
      - a custom input model provided in the request body.
    """
    if archetype:
        if not category or not country or not name:
            raise HTTPException(status_code=400, detail="With archetype=true you must provide 'category', 'country' and 'name'.")

        match = next(
            (b for b in BUILDING_ARCHETYPES if b.get("category") == category and b.get("country") == country and b.get("name") == name),
            None,
        )
        if match is None:
            raise HTTPException(status_code=404, detail=f"No archetype found for category='{category}', country='{country}', name='{name}'.")

        bui_internal = match["bui"]
        system_internal = match["system"]

    else:
        if payload is None:
            raise HTTPException(status_code=400, detail="With archetype=false you must send a JSON body containing 'bui' and 'system'.")

        bui_json = payload.get("bui")
        system_json = payload.get("system")
        if bui_json is None or system_json is None:
            raise HTTPException(status_code=400, detail="Incomplete JSON body: both 'bui' and 'system' are required.")

        bui_internal = json_to_internal_bui(bui_json)
        system_internal = json_to_internal_system(system_json)

    result = validate_bui_and_system(bui_internal, system_internal)

    return {
        "source": "archetype" if archetype else "custom",
        "name": name,
        "category": category,
        "country": country,
        "bui_fixed": clean_and_jsonable(result["bui_fixed"]),
        "bui_checked": clean_and_jsonable(result["bui_checked"]),
        "system_checked": clean_and_jsonable(result["system_checked"]),
        "bui_report_fixed": clean_and_jsonable(result["bui_report_fixed"]),
        "bui_issues": result["bui_issues"],
        "system_messages": result["system_messages"],
    }


# =============================================================================
# Simulation endpoints
# =============================================================================

@app.post("/simulate", tags=["Simulation"])
async def simulate_building(
    archetype: bool = Query(True, description="If True, simulate an archetype; if False, use custom BUI/SYSTEM."),
    category: Optional[str] = Query(None, description="Building category. Required when archetype=True."),
    country: Optional[str] = Query(None, description="Country. Required when archetype=True."),
    name: Optional[str] = Query(None, description="Archetype name. Required when archetype=True."),
    weather_source: str = Query("pvgis", description="Weather source: 'pvgis' or 'epw'."),
    epw_file: Optional[UploadFile] = File(None, description="EPW weather file (required when weather_source='epw')."),
    bui_json: Optional[str] = Form(None, description="JSON string with the BUI data (required when archetype=False)."),
    system_json: Optional[str] = Form(None, description="JSON string with the SYSTEM data (required when archetype=False)."),
):
    """
    Simulate a single building (ISO 52016 + ISO 15316) using either:
      - an archetype (archetype=True), or
      - a custom BUI/system configuration (archetype=False).

    Weather:
      - weather_source='pvgis' -> data retrieved from PVGIS
      - weather_source='epw'   -> EPW is uploaded via epw_file
    """
    # 1) Select inputs
    if archetype:
        if not category or not country or not name:
            raise HTTPException(status_code=400, detail="With archetype=true you must provide 'category', 'country' and 'name'.")

        match = next(
            (b for b in BUILDING_ARCHETYPES if b.get("category") == category and b.get("country") == country and b.get("name") == name),
            None,
        )
        if match is None:
            raise HTTPException(status_code=404, detail=f"No archetype found for category='{category}', country='{country}', name='{name}'.")

        bui_internal = match["bui"]
        system_internal = match["system"]

    else:
        if bui_json is None or system_json is None:
            raise HTTPException(status_code=400, detail="With archetype=false you must send 'bui_json' and 'system_json' as form fields.")

        try:
            bui_raw = json.loads(bui_json)
            system_raw = json.loads(system_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="bui_json and system_json must be valid JSON strings.")

        bui_internal = json_to_internal_bui(bui_raw)
        system_internal = json_to_internal_system(system_raw)

    # 2) Validate inputs
    validation = validate_bui_and_system(bui_internal, system_internal)
    bui_checked = validation["bui_checked"]
    system_checked = validation["system_checked"]

    # 3) Weather + ISO 52016
    if weather_source == "pvgis":

        @retry_on_transient_error()
        def _run_pvgis_simulation():
            return pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                bui_checked,
                weather_source="pvgis",
                sankey_graph=False,
            )

        hourly_sim, annual_results_df = _run_pvgis_simulation()

    elif weather_source == "epw":
        if epw_file is None:
            raise HTTPException(status_code=400, detail="With weather_source='epw' you must upload 'epw_file'.")

        with tempfile.TemporaryDirectory() as tmpdir:
            epw_path = os.path.join(tmpdir, epw_file.filename)
            contents = await epw_file.read()
            with open(epw_path, "wb") as f:
                f.write(contents)

            hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                bui_checked,
                weather_source="epw",
                path_weather_file=epw_path,
                sankey_graph=False,
            )
    else:
        raise HTTPException(status_code=400, detail="weather_source must be either 'pvgis' or 'epw'.")

    # 4) ISO 15316 heating system simulation
    calc = pybui.HeatingSystemCalculator(system_checked)
    calc.load_csv_data(hourly_sim)
    df_system = calc.run_timeseries()

    # 5) UNI/TS 11300 primary energy (from hourly ideal loads)
    uni11300_results = _compute_uni11300_from_hourly_df(hourly_sim)

    # 6) Response
    return {
        "source": "archetype" if archetype else "custom",
        "name": name,
        "category": category,
        "country": country,
        "weather_source": weather_source,
        "validation": {
            "bui_issues": validation["bui_issues"],
            "system_messages": validation["system_messages"],
        },
        "results": {
            "hourly_building": dataframe_to_records_safe(hourly_sim),
            "primary_energy_uni11300": uni11300_results,
            # keep commented if you still want to reduce payload size:
            # "annual_building": dataframe_to_records_safe(annual_results_df),
            # "hourly_system": dataframe_to_records_safe(df_system),
        },
    }


# =============================================================================
# Primary energy (UNI/TS 11300) endpoints
# =============================================================================

UNI11300_INPUT_EXAMPLE = build_uni11300_input_example()


def _compute_uni11300_from_hourly_df(hourly_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute UNI/TS 11300 results from a pybuildingenergy hourly DataFrame.

    Assumptions:
      - Q_H and Q_C are in Wh (as returned by /simulate)
    """
    if hourly_df is None or hourly_df.empty:
        raise HTTPException(status_code=400, detail="Hourly simulation data is empty.")

    heat_series = None
    cool_series = None
    if "Q_H" in hourly_df.columns:
        heat_series = hourly_df["Q_H"].astype(float)
    elif "Q_HC" in hourly_df.columns:
        heat_series = hourly_df["Q_HC"].clip(lower=0.0).astype(float)

    if "Q_C" in hourly_df.columns:
        cool_series = hourly_df["Q_C"].astype(float)
    elif "Q_HC" in hourly_df.columns:
        cool_series = (-hourly_df["Q_HC"]).clip(lower=0.0).astype(float)

    if heat_series is None and cool_series is None:
        raise HTTPException(status_code=400, detail="Hourly data must include Q_H/Q_C (or Q_HC).")

    df_ideal = pd.DataFrame(index=hourly_df.index)
    if heat_series is not None:
        df_ideal["Q_ideal_heat_kWh"] = heat_series * 0.001
    if cool_series is not None:
        df_ideal["Q_ideal_cool_kWh"] = cool_series * 0.001

    results_df = compute_primary_energy_from_hourly_ideal(
        df_hourly=df_ideal,
        heat_col="Q_ideal_heat_kWh",
        cool_col="Q_ideal_cool_kWh",
        heating_params=HeatingSystemParams(),
        cooling_params=CoolingSystemParams(),
    )

    summary_fields = [
        "Q_ideal_heat_kWh",
        "Q_ideal_cool_kWh",
        "E_delivered_thermal_kWh",
        "E_delivered_electric_heat_kWh",
        "E_delivered_electric_cool_kWh",
        "E_delivered_electric_total_kWh",
        "EP_heat_total_kWh",
        "EP_cool_total_kWh",
        "EP_total_kWh",
    ]
    summary = {col: float(results_df[col].sum()) for col in summary_fields if col in results_df.columns}

    return {
        "input_unit": "Wh",
        "ideal_unit": "kWh",
        "n_hours": len(results_df),
        "summary": summary,
    }


def _extract_hourly_records_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accept either a direct hourly list or a full /simulate response.

    Supported shapes:
      - {"hourly_sim": [...]}
      - {"hourly_building": [...]}
      - {"results": {"hourly_building": [...]}}
      - {"results": {"hourly_building": {"hourly_sim": [...]}}}
      - {"simulate_result": <any of the above>}
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object.")

    candidate = None
    if "hourly_sim" in payload:
        candidate = payload.get("hourly_sim")
    elif "hourly_building" in payload:
        candidate = payload.get("hourly_building")
    elif isinstance(payload.get("results"), dict):
        hourly_building = payload["results"].get("hourly_building")
        if isinstance(hourly_building, dict) and "hourly_sim" in hourly_building:
            candidate = hourly_building.get("hourly_sim")
        else:
            candidate = hourly_building
    elif "simulate_result" in payload:
        return _extract_hourly_records_from_payload(payload["simulate_result"])

    if candidate is None:
        raise HTTPException(
            status_code=400,
            detail="Missing hourly data. Provide 'hourly_sim', 'hourly_building', or a full /simulate response.",
        )

    if not isinstance(candidate, list):
        raise HTTPException(status_code=400, detail="Hourly data must be a list of records.")

    return candidate


@app.get("/primary-energy/uni11300/input-example", tags=["Primary Energy"])
def get_uni11300_input_example():
    """
    Return an example input payload for the UNI/TS 11300 calculation.
    """
    return UNI11300_INPUT_EXAMPLE


@app.post("/primary-energy/uni11300", tags=["Primary Energy"])
def compute_primary_energy_uni11300(payload: Dict[str, Any]):
    """
    Compute delivered and primary energy (UNI/TS 11300) from hourly ideal loads.

    Input:
      - Use /simulate output (Q_H and Q_C columns) or pass hourly records directly.
      - Q_H -> ideal heating load, Q_C -> ideal cooling load.
      - Default input_unit is "Wh" (as returned by /simulate); set to "kWh" if already in kWh.
      - Optional "heating_params" and "cooling_params" override defaults.
    """
    hourly_records = _extract_hourly_records_from_payload(payload)
    hourly_df = pd.DataFrame(hourly_records)
    if hourly_df.empty:
        raise HTTPException(status_code=400, detail="Hourly data is empty.")

    if "timestamp" in hourly_df.columns:
        hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"], errors="coerce")
        if hourly_df["timestamp"].isna().any():
            raise HTTPException(status_code=400, detail="Invalid timestamps in hourly data.")
        hourly_df = hourly_df.set_index("timestamp").sort_index()

    input_unit = str(payload.get("input_unit") or "Wh").strip().lower()
    if input_unit not in {"wh", "kwh"}:
        raise HTTPException(status_code=400, detail="input_unit must be 'Wh' or 'kWh'.")
    scale_to_kwh = 0.001 if input_unit == "wh" else 1.0

    heat_series = None
    cool_series = None
    if "Q_H" in hourly_df.columns:
        heat_series = hourly_df["Q_H"].astype(float)
    elif "Q_HC" in hourly_df.columns:
        heat_series = hourly_df["Q_HC"].clip(lower=0.0).astype(float)

    if "Q_C" in hourly_df.columns:
        cool_series = hourly_df["Q_C"].astype(float)
    elif "Q_HC" in hourly_df.columns:
        cool_series = (-hourly_df["Q_HC"]).clip(lower=0.0).astype(float)

    if heat_series is None and cool_series is None:
        raise HTTPException(status_code=400, detail="No Q_H/Q_C (or Q_HC) columns found in hourly data.")

    df_ideal = pd.DataFrame(index=hourly_df.index)
    if heat_series is not None:
        df_ideal["Q_ideal_heat_kWh"] = heat_series * scale_to_kwh
    if cool_series is not None:
        df_ideal["Q_ideal_cool_kWh"] = cool_series * scale_to_kwh

    heating_params_payload = payload.get("heating_params") or {}
    cooling_params_payload = payload.get("cooling_params") or {}

    if not isinstance(heating_params_payload, dict):
        raise HTTPException(status_code=400, detail="'heating_params' must be a JSON object.")
    if not isinstance(cooling_params_payload, dict):
        raise HTTPException(status_code=400, detail="'cooling_params' must be a JSON object.")

    try:
        heating_params = HeatingSystemParams(**heating_params_payload)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid heating_params: {exc}") from exc

    try:
        cooling_params = CoolingSystemParams(**cooling_params_payload)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid cooling_params: {exc}") from exc

    results_df = compute_primary_energy_from_hourly_ideal(
        df_hourly=df_ideal,
        heat_col="Q_ideal_heat_kWh",
        cool_col="Q_ideal_cool_kWh",
        heating_params=heating_params,
        cooling_params=cooling_params,
    )

    summary_fields = [
        "Q_ideal_heat_kWh",
        "Q_ideal_cool_kWh",
        "E_delivered_thermal_kWh",
        "E_delivered_electric_heat_kWh",
        "E_delivered_electric_cool_kWh",
        "E_delivered_electric_total_kWh",
        "EP_heat_total_kWh",
        "EP_cool_total_kWh",
        "EP_total_kWh",
    ]
    summary = {col: float(results_df[col].sum()) for col in summary_fields if col in results_df.columns}

    return {
        "input_unit": input_unit,
        "ideal_unit": "kWh",
        "n_hours": len(results_df),
        "hourly_results": dataframe_to_records_safe(results_df),
        "summary": summary,
    }


@app.post("/simulate/batch", tags=["Simulation"])
def simulate_batch(payload: Dict[str, Any]):
    """
    Batch-simulate multiple buildings in parallel (multiprocessing).

    Modes:
      - mode="archetype": filter BUILDING_ARCHETYPES by category/countries/names
      - mode="custom": accept a list of {name, bui, system}
    """
    mode = payload.get("mode")
    buildings_to_simulate: List[Dict[str, Any]] = []

    if mode == "archetype":
        category = payload.get("category")
        countries = payload.get("countries") or []
        names = payload.get("names") or []

        if not category or not countries or not names:
            raise HTTPException(status_code=400, detail="For mode='archetype' provide 'category', 'countries' (list) and 'names' (list).")

        for arch in BUILDING_ARCHETYPES:
            if arch.get("category") == category and arch.get("country") in countries and arch.get("name") in names:
                buildings_to_simulate.append(
                    {"name": arch["name"], "bui": arch["bui"], "system": arch["system"], "category": arch.get("category"), "country": arch.get("country")}
                )

        if not buildings_to_simulate:
            raise HTTPException(status_code=404, detail="No archetypes found for the given parameters.")

    elif mode == "custom":
        buildings = payload.get("buildings")
        if not buildings:
            raise HTTPException(status_code=400, detail="For mode='custom' provide 'buildings': [ {name,bui,system} ].")

        for b in buildings:
            bname = b.get("name")
            bui_json = b.get("bui")
            system_json = b.get("system")

            if not bname or bui_json is None or system_json is None:
                raise HTTPException(status_code=400, detail="Each custom building must have 'name', 'bui' and 'system'.")

            buildings_to_simulate.append(
                {
                    "name": bname,
                    "bui": json_to_internal_bui(bui_json),
                    "system": json_to_internal_system(system_json),
                    "category": b.get("category"),
                    "country": b.get("country"),
                }
            )
    else:
        raise HTTPException(status_code=400, detail="Field 'mode' must be either 'archetype' or 'custom'.")

    results: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(simulate_building_worker, b["name"], b["bui"], b["system"])
            for b in buildings_to_simulate
        ]

        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            if res.get("status") == "ok":
                summaries.append(res["summary"])

    summary_df = pd.DataFrame(summaries) if summaries else pd.DataFrame()

    return {
        "status": "completed",
        "mode": mode,
        "n_buildings": len(results),
        "results": results,
        "summary": summary_df.to_dict(orient="records"),
    }


# =============================================================================
# Report endpoint
# =============================================================================

@app.post("/report", response_class=HTMLResponse, tags=["Reports"])
def generate_report(payload: Dict[str, Any]):
    """
    Generate an HTML report with hourly/monthly/annual statistical analysis of energy needs.

    Input JSON must contain:
      - hourly_sim: list of records representing the hourly DataFrame
      - building_area: float (m²)
    """
    hourly_records = payload.get("hourly_sim")
    building_area = payload.get("building_area")

    if hourly_records is None or building_area is None:
        raise HTTPException(status_code=400, detail="Request JSON must contain 'hourly_sim' and 'building_area'.")

    hourly_sim = pd.DataFrame(hourly_records)

    with tempfile.TemporaryDirectory() as tmpdir:
        name_report = "energy_report"

        report_obj = pybui.Graphs_and_report(
            df=hourly_sim,
            season="heating_cooling",
            building_area=building_area,
        )
        report_obj.bui_analysis_page(folder_directory=tmpdir, name_file=name_report)

        html_path = os.path.join(tmpdir, f"{name_report}.html")
        if not os.path.exists(html_path):
            raise HTTPException(status_code=500, detail="HTML report not found. Check 'bui_analysis_page' output.")

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

    return HTMLResponse(content=html_content)


# =============================================================================
# Greek EPC endpoint
# =============================================================================

@app.post("/bui/epc_update_u_values", tags=["Greek EPC"])
def update_u_values(
    energy_class: str = Query(..., description="Energy class (A, B, C, D) for greek buildings", regex="^[ABCD]$"),
    archetype: bool = Query(True, description="If True, load from archetype and create a NEW modified archetype. If False, modify a custom BUI (not stored)."),
    category: Optional[str] = Query(None, description="Building category. Required when archetype=True."),
    country: Optional[str] = Query(None, description="Country. Required when archetype=True."),
    name: Optional[str] = Query(None, description="Name of the source archetype. Required when archetype=True."),
    new_name: Optional[str] = Query(None, description="Name of the NEW archetype to be created (optional)."),
    u_slab: Optional[float] = Query(None, description="Override U-value for slab to ground (optional)."),
    use_heat_pump: bool = Query(False, description="If True, switch generation to heat pump (system update)."),
    heat_pump_cop: float = Query(3.2, description="Heat pump COP used for system update (if enabled)."),
    payload: Optional[Dict[str, Any]] = Body(None, description="JSON body with 'bui' (and optionally 'system') when archetype=False."),
):
    """
    Update envelope U-values in the BUI according to an energy class (A/B/C/D).

    Behavior:
      - archetype=True: creates a NEW archetype and appends it to BUILDING_ARCHETYPES.
      - archetype=False: modifies a provided BUI and returns it (no persistence).

    Optional:
      - u_slab: override slab-to-ground U-value after class mapping
      - use_heat_pump: update system to a heat pump generator (requires system)
    """
    base_system = None
    base_name = name

    if archetype:
        if not category or not country or not name:
            raise HTTPException(status_code=400, detail="With archetype=true you must provide 'category', 'country' and 'name'.")

        match = next(
            (b for b in BUILDING_ARCHETYPES if b.get("category") == category and b.get("country") == country and b.get("name") == name),
            None,
        )
        if match is None:
            raise HTTPException(status_code=404, detail=f"No archetype found for category='{category}', country='{country}', name='{name}'.")

        bui_internal = copy.deepcopy(match["bui"])
        base_system = copy.deepcopy(match["system"])

    else:
        if payload is None:
            raise HTTPException(status_code=400, detail="With archetype=false you must send a JSON body with at least 'bui'.")

        bui_json = payload.get("bui")
        system_json = payload.get("system")
        if bui_json is None:
            raise HTTPException(status_code=400, detail="Incomplete JSON body: missing 'bui' key.")

        bui_internal = json_to_internal_bui(bui_json)
        if system_json is not None:
            base_system = json_to_internal_system(system_json)

    class_map = U_VALUES_BY_CLASS.get(energy_class)
    if class_map is None:
        raise HTTPException(status_code=400, detail=f"Invalid energy class '{energy_class}'. Use A, B, C or D.")

    for surf in bui_internal.get("building_surface", []):
        name_surf = surf.get("name")
        if not name_surf:
            continue
        key = _norm_surface_name(name_surf)
        new_u = class_map.get(key)
        if new_u is not None:
            surf["u_value"] = float(new_u)

    if u_slab is not None:
        for surf in bui_internal.get("building_surface", []):
            if classify_surface(surf) == "slab":
                surf["u_value"] = float(u_slab)

    if use_heat_pump:
        if heat_pump_cop <= 0:
            raise HTTPException(status_code=400, detail="heat_pump_cop must be > 0.")
        if base_system is None:
            raise HTTPException(
                status_code=400,
                detail="use_heat_pump=true requires a system in the archetype or in the request payload.",
            )
        base_system = apply_heat_pump_to_system(base_system, cop=heat_pump_cop)

    created_archetype_name = None
    if archetype:
        created_archetype_name = new_name or f"{base_name}_class_{energy_class}"
        BUILDING_ARCHETYPES.append(
            {
                "name": created_archetype_name,
                "category": category,
                "country": country,
                "bui": bui_internal,
                "system": base_system,
            }
        )

    return {
        "source": "archetype" if archetype else "custom",
        "base_name": base_name,
        "base_category": category,
        "base_country": country,
        "energy_class": energy_class,
        "new_archetype_name": created_archetype_name,
        "bui": to_jsonable(bui_internal),
        "system": to_jsonable(base_system) if base_system is not None else None,
    }


# =============================================================================
# ECM endpoint (U-values scenario simulation)
# =============================================================================
from typing import Set

@app.post("/ecm_application", tags=["Simulation"])
async def simulate_uvalues(
    archetype: bool = Query(True, description="If True, use an archetype; if False, use a custom BUI from JSON."),
    category: Optional[str] = Query(None, description="Building category. Required when archetype=True."),
    country: Optional[str] = Query(None, description="Country. Required when archetype=True."),
    name: Optional[str] = Query(None, description="Archetype name. Required when archetype=True."),
    weather_source: str = Query("pvgis", description="Weather source: 'pvgis' or 'epw'."),
    epw_file: Optional[UploadFile] = File(None, description="EPW weather file (required when weather_source='epw')."),
    bui_json: Optional[str] = Form(None, description="JSON string with the BUI (required when archetype=False)."),
    system_json: Optional[str] = Form(None, description="JSON string with SYSTEM (required when use_heat_pump=true in custom mode)."),
    u_wall: Optional[float] = Query(None, description="New wall U-value (opaque vertical surfaces)."),
    u_roof: Optional[float] = Query(None, description="New roof U-value (opaque horizontal surface)."),
    u_window: Optional[float] = Query(None, description="New window U-value (transparent vertical surfaces)."),
    u_slab: Optional[float] = Query(None, description="New slab-to-ground U-value (opaque ground surface)."),
    use_heat_pump: bool = Query(False, description="If True, switch generation to heat pump (system update)."),
    heat_pump_cop: float = Query(3.2, description="Heat pump COP used for system update (if enabled)."),

    # ---------------------------
    # NEW: single scenario mode
    # ---------------------------
    scenario_elements: Optional[str] = Query(
        None,
        description="Single scenario mode: elements to apply, e.g. 'wall,window' or 'roof+wall'.",
    ),
    scenario_id: Optional[str] = Query(
        None,
        description="Single scenario mode: pick a scenario by id returned by build_uvalue_scenarios().",
    ),
    baseline_only: bool = Query(
        False,
        description="If True, simulate only baseline (no envelope changes), ignoring scenario_elements/scenario_id.",
    ),

    # (optional) keep your previous flag if you added it
    include_baseline: bool = Query(False, description="If True, also simulate baseline (no changes)."),
):
    """
    Build all non-empty combinations of the requested U-value interventions and simulate each scenario.

    Single scenario mode:
      - baseline_only=true -> baseline only
      - scenario_id=<id> OR scenario_elements=wall,window -> only that scenario

    Optional:
      - u_slab: slab-to-ground U-value
      - use_heat_pump: update system to heat pump (returned as system_variant)
    """

    # 1) Base BUI
    base_system = None
    if archetype:
        if not category or not country or not name:
            raise HTTPException(status_code=400, detail="With archetype=true you must provide 'category', 'country' and 'name'.")
        match = next(
            (b for b in BUILDING_ARCHETYPES if b.get("category") == category and b.get("country") == country and b.get("name") == name),
            None,
        )
        if match is None:
            raise HTTPException(status_code=404, detail=f"No archetype found for category='{category}', country='{country}', name='{name}'.")
        base_bui = match["bui"]
        base_system = match.get("system")
    else:
        if bui_json is None:
            raise HTTPException(status_code=400, detail="With archetype=false you must send 'bui_json' as a form field.")
        try:
            base_bui = json.loads(bui_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="'bui_json' must be a valid JSON string.")
        if system_json is not None:
            try:
                system_raw = json.loads(system_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="'system_json' must be a valid JSON string.")
            base_system = json_to_internal_system(system_raw)

    # 2) Build scenarios (as before)
    scenarios_spec = build_uvalue_scenarios(u_roof=u_roof, u_wall=u_wall, u_window=u_window, u_slab=u_slab)

    # --- NEW: parse scenario_elements helper
    def _parse_elements(s: str) -> Set[str]:
        # accept separators: comma, plus, space, semicolon
        raw = s.replace("+", ",").replace(";", ",").replace(" ", ",")
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        mapping = {"ground": "slab"}
        normalized = [mapping.get(p, p) for p in parts]
        allowed = {"roof", "wall", "window", "slab"}
        bad = [p for p in normalized if p not in allowed]
        if bad:
            raise HTTPException(status_code=400, detail=f"Invalid scenario_elements: {bad}. Allowed: roof, wall, window, slab.")
        return set(normalized)

    # --- NEW: "single scenario mode" filter
    if baseline_only:
        # simulate baseline only -> no ECM scenarios
        scenarios_spec = []
        include_baseline = True  # force baseline result

    else:
        # If user asked for one scenario, filter scenarios_spec down to 1
        if scenario_id:
            filtered = [s for s in scenarios_spec if str(s.get("id")) == str(scenario_id)]
            if not filtered:
                raise HTTPException(
                    status_code=404,
                    detail=f"scenario_id='{scenario_id}' not found. Available ids: {[s.get('id') for s in scenarios_spec]}",
                )
            scenarios_spec = filtered

        elif scenario_elements:
            wanted = _parse_elements(scenario_elements)

            def _spec_to_elements(spec: Dict[str, Any]) -> Set[str]:
                els: Set[str] = set()
                if spec.get("use_roof"):
                    els.add("roof")
                if spec.get("use_wall"):
                    els.add("wall")
                if spec.get("use_window"):
                    els.add("window")
                if spec.get("use_slab"):
                    els.add("slab")
                return els

            filtered = [s for s in scenarios_spec if _spec_to_elements(s) == wanted]
            if not filtered:
                available = []
                for s in scenarios_spec:
                    available.append(
                        {
                            "id": s.get("id"),
                            "elements": sorted(list(_spec_to_elements(s))),
                            "label": s.get("label"),
                        }
                    )
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": f"scenario_elements='{scenario_elements}' not found among generated scenarios.",
                        "available": available,
                    },
                )
            scenarios_spec = filtered

    # 3) Validate we have something to do
    if not scenarios_spec and not include_baseline:
        raise HTTPException(
            status_code=400,
            detail="Nothing to simulate: specify at least one of u_wall/u_roof/u_window/u_slab, or set include_baseline=true, or baseline_only=true.",
        )

    system_variant = None
    if use_heat_pump:
        if heat_pump_cop <= 0:
            raise HTTPException(status_code=400, detail="heat_pump_cop must be > 0.")
        if base_system is None:
            raise HTTPException(
                status_code=400,
                detail="use_heat_pump=true requires a system in the archetype or in the request payload.",
            )
        system_variant = apply_heat_pump_to_system(base_system, cop=heat_pump_cop)

    # 4) Weather handling for EPW (as before)
    epw_path: Optional[str] = None
    if weather_source == "epw":
        if epw_file is None:
            raise HTTPException(status_code=400, detail="With weather_source='epw' you must upload 'epw_file'.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epw") as tmp:
            tmp.write(await epw_file.read())
            epw_path = tmp.name
    elif weather_source != "pvgis":
        raise HTTPException(status_code=400, detail="weather_source must be 'pvgis' or 'epw'.")

    scenario_results: List[Dict[str, Any]] = []
    try:
        # 5) Baseline (optional)
        if include_baseline:
            if weather_source == "pvgis":

                @retry_on_transient_error()
                def _run_baseline_pvgis():
                    return pybui.ISO52016.Temperature_and_Energy_needs_calculation(base_bui, weather_source="pvgis", sankey_graph=False,)

                hourly_sim, annual_results_df = _run_baseline_pvgis()
            else:
                hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                    base_bui, weather_source="epw", path_weather_file=epw_path, sankey_graph=False, 
                )

            scenario_results.append(
                {
                    "scenario_id": "baseline",
                    "description": "Baseline (no changes)",
                    "elements": [],
                    "u_values": {"roof": None, "wall": None, "window": None, "slab": None},
                    "results": {
                        "hourly_building": clean_and_jsonable(hourly_sim),
                        "annual_building": clean_and_jsonable(annual_results_df),
                    },
                }
            )

        # 6) ECM scenarios (now possibly filtered to 1)
        for spec in scenarios_spec:
            elements = []
            if spec.get("use_roof"):
                elements.append("roof")
            if spec.get("use_wall"):
                elements.append("wall")
            if spec.get("use_window"):
                elements.append("window")
            if spec.get("use_slab"):
                elements.append("slab")

            bui_variant = apply_u_values_to_bui(
                base_bui,
                use_roof=spec["use_roof"],
                use_wall=spec["use_wall"],
                use_window=spec["use_window"],
                use_slab=spec["use_slab"],
                u_roof=u_roof,
                u_wall=u_wall,
                u_window=u_window,
                u_slab=u_slab,
            )

            if weather_source == "pvgis":

                @retry_on_transient_error()
                def _run_scenario_pvgis():
                    return pybui.ISO52016.Temperature_and_Energy_needs_calculation(bui_variant, weather_source="pvgis", sankey_graph=False,)

                hourly_sim, annual_results_df = _run_scenario_pvgis()
            else:
                hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                    bui_variant, weather_source="epw", path_weather_file=epw_path, sankey_graph=False,
                )

            scenario_results.append(
                {
                    "scenario_id": spec["id"],
                    "description": spec["label"],
                    "elements": elements,
                    "u_values": {
                        "roof": (u_roof if "roof" in elements else None),
                        "wall": (u_wall if "wall" in elements else None),
                        "window": (u_window if "window" in elements else None),
                        "slab": (u_slab if "slab" in elements else None),
                    },
                    "results": {
                        "hourly_building": clean_and_jsonable(hourly_sim),
                        "annual_building": clean_and_jsonable(annual_results_df),
                    },
                }
            )

    finally:
        if epw_path and os.path.exists(epw_path):
            try:
                os.remove(epw_path)
            except OSError:
                pass

    return {
        "source": "archetype" if archetype else "custom",
        "name": name,
        "category": category,
        "country": country,
        "weather_source": weather_source,
        "u_values_requested": {"roof": u_roof, "wall": u_wall, "window": u_window, "slab": u_slab},
        "system_variant": to_jsonable(system_variant) if system_variant is not None else None,
        "single_scenario_mode": {
            "baseline_only": baseline_only,
            "scenario_id": scenario_id,
            "scenario_elements": scenario_elements,
        },
        "n_scenarios": len(scenario_results),
        "scenarios": scenario_results,
    }




# -----------------------------------------------------------------------------
# Helpers (safe + minimal)
# -----------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return (
        str(text).lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _ensure_dir(p: Union[str, Path]) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    return obj


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge two dictionaries without mutating inputs.
    Values from `override` win over `base`.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build_name_file(
    building_name: str,
    combo: List[str],
    weather_source: str,
    epw_path: Optional[Union[str, Path]],
    base_dir: Union[str, Path],
) -> str:
    base_dir = _ensure_dir(base_dir)
    bld = _slugify(building_name)
    ecm_tag = "_".join(sorted(combo)) if combo else "baseline"
    weather_tag = "pvgis"
    if weather_source == "epw" and epw_path:
        weather_tag = _slugify(Path(epw_path).stem)
    filename = f"{bld}__{ecm_tag}__{weather_tag}.csv"
    return str(base_dir / filename)


def _save_bui_variant(
    bui_obj: Dict[str, Any],
    active_elements: List[str],
    bui_dir: Union[str, Path],
) -> str:
    _ensure_dir(bui_dir)
    building_name = bui_obj.get("building", {}).get("name", "building")
    combo_tag = "_".join(sorted(active_elements)) if active_elements else "baseline"
    filename = f"BUI_{_slugify(building_name)}__{combo_tag}.json"
    full_path = Path(bui_dir) / filename
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(bui_obj), f, ensure_ascii=False, indent=2)
    return str(full_path)


def _generate_combinations(ecm_options: List[str], include_baseline: bool) -> List[List[str]]:
    combos: List[List[str]] = []
    if include_baseline:
        combos.append([])
    for r in range(1, len(ecm_options) + 1):
        for subset in itertools.combinations(ecm_options, r):
            combos.append(list(subset))
    return combos


# -----------------------------------------------------------------------------
# NEW ENDPOINT: sequential run + save CSV (no multiprocessing)
# -----------------------------------------------------------------------------
@app.post("/ecm_application/run_sequential_save", tags=["Simulation"])
async def ecm_application_run_sequential_save(
    # Input selection (same as /ecm_application)
    archetype: bool = Query(True, description="If True, use an archetype; if False, use a custom BUI from JSON."),
    category: Optional[str] = Query(None, description="Building category. Required when archetype=True."),
    country: Optional[str] = Query(None, description="Country. Required when archetype=True."),
    name: Optional[str] = Query(None, description="Archetype name. Required when archetype=True."),
    weather_source: str = Query("pvgis", description="Weather source: 'pvgis' or 'epw'."),
    epw_file: Optional[UploadFile] = File(None, description="EPW weather file (required when weather_source='epw')."),

    # ECM controls
    ecm_options: Optional[str] = Query(
        None,
        description="Comma-separated list of elements to combine, e.g. 'wall,window,roof'. "
                    "If omitted, it is inferred from which u_* are provided.",
    ),
    u_wall: Optional[float] = Query(None, description="New wall U-value."),
    u_roof: Optional[float] = Query(None, description="New roof U-value."),
    u_window: Optional[float] = Query(None, description="New window U-value."),
    u_slab: Optional[float] = Query(None, description="New slab-to-ground U-value."),
    include_baseline: bool = Query(True, description="If True, also run baseline (no changes)."),

    # Saving controls
    output_dir: str = Query("results/ecm_api", description="Folder where CSV results are saved."),
    save_bui: bool = Query(True, description="If True, save BUI JSON variants on disk."),
    bui_dir: str = Query("building_examples_ecm_api", description="Folder where BUI JSON variants are saved."),
) -> Dict[str, Any]:
    """
    Runs all combinations sequentially (NO multiprocessing) and saves:
      - hourly CSV
      - annual CSV
      - optional BUI JSON per scenario

    This endpoint does NOT return huge hourly payloads; it returns file paths + status.
    """

    # -------------------------
    # 1) Resolve base_bui
    # -------------------------
    if archetype:
        if not category or not country or not name:
            raise HTTPException(status_code=400, detail="With archetype=true you must provide 'category', 'country' and 'name'.")

        match = next(
            (b for b in BUILDING_ARCHETYPES if b.get("category") == category and b.get("country") == country and b.get("name") == name),
            None,
        )
        if match is None:
            raise HTTPException(status_code=404, detail=f"No archetype found for category='{category}', country='{country}', name='{name}'.")
        base_bui = match["bui"]
        building_name = match.get("name", name) or "building"
    else:
        # If you want custom mode too, you can extend the signature with bui_json=Form(...)
        raise HTTPException(status_code=400, detail="This endpoint currently supports archetype=true only (extend it if needed).")

    # -------------------------
    # 2) Determine ecm_options list
    # -------------------------
    if ecm_options:
        opts = [x.strip().lower() for x in ecm_options.split(",") if x.strip()]
    else:
        # infer from provided u-values
        opts = []
        if u_wall is not None:
            opts.append("wall")
        if u_roof is not None:
            opts.append("roof")
        if u_window is not None:
            opts.append("window")
        if u_slab is not None:
            opts.append("slab")

    allowed = {"wall", "roof", "window", "slab"}
    bad = [o for o in opts if o not in allowed]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid ecm_options={bad}. Allowed: wall, roof, window, slab.")
    if not opts and not include_baseline:
        raise HTTPException(status_code=400, detail="Nothing to run: provide ecm_options/u-values or set include_baseline=true.")

    # validations similar to your client
    if "wall" in opts and u_wall is None:
        raise HTTPException(status_code=400, detail="For option 'wall' you must provide u_wall.")
    if "roof" in opts and u_roof is None:
        raise HTTPException(status_code=400, detail="For option 'roof' you must provide u_roof.")
    if "window" in opts and u_window is None:
        raise HTTPException(status_code=400, detail="For option 'window' you must provide u_window.")
    if "slab" in opts and u_slab is None:
        raise HTTPException(status_code=400, detail="For option 'slab' you must provide u_slab.")

    combos = _generate_combinations(opts, include_baseline=include_baseline)

    # -------------------------
    # 3) Weather EPW handling (single temp file)
    # -------------------------
    epw_path: Optional[str] = None
    if weather_source == "epw":
        if epw_file is None:
            raise HTTPException(status_code=400, detail="With weather_source='epw' you must upload 'epw_file'.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epw") as tmp:
            tmp.write(await epw_file.read())
            epw_path = tmp.name
    elif weather_source != "pvgis":
        raise HTTPException(status_code=400, detail="weather_source must be 'pvgis' or 'epw'.")

    # -------------------------
    # 4) Run sequentially + save
    # -------------------------
    _ensure_dir(output_dir)
    if save_bui:
        _ensure_dir(bui_dir)

    results: List[Dict[str, Any]] = []
    start_all = time.time()

    try:
        for combo in combos:
            t0 = time.time()
            combo_tag = ",".join(combo) if combo else "BASELINE"

            try:
                # Apply U-values using YOUR existing helper from forecasting_service_functions
                # (same one used inside /ecm_application)
                bui_variant = apply_u_values_to_bui(
                    base_bui,
                    use_roof=("roof" in combo),
                    use_wall=("wall" in combo),
                    use_window=("window" in combo),
                    use_slab=("slab" in combo),
                    u_roof=u_roof,
                    u_wall=u_wall,
                    u_window=u_window,
                    u_slab=u_slab,
                ) if combo else base_bui

                if save_bui:
                    bui_json_path = _save_bui_variant(bui_variant, combo, bui_dir)
                else:
                    bui_json_path = None

                # simulate (ISO52016)
                if weather_source == "pvgis":

                    @retry_on_transient_error()
                    def _run_sequential_pvgis():
                        return pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                            bui_variant,
                            weather_source="pvgis",
                            sankey_graph=False,
                        )

                    hourly_sim, annual_results_df = _run_sequential_pvgis()
                else:
                    hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                        bui_variant,
                        weather_source="epw",
                        path_weather_file=epw_path,
                        sankey_graph=False,
                    )

                # save csv
                hourly_csv = _build_name_file(
                    building_name=building_name,
                    combo=combo,
                    weather_source=weather_source,
                    epw_path=epw_path,
                    base_dir=output_dir,
                )
                pd.DataFrame(hourly_sim).to_csv(hourly_csv, index=False)

                annual_csv = str(Path(hourly_csv).with_name(Path(hourly_csv).stem + "__annual.csv"))
                pd.DataFrame(annual_results_df).to_csv(annual_csv, index=False)

                results.append(
                    {
                        "status": "success",
                        "combo": combo,
                        "combo_tag": combo_tag,
                        "files": {
                            "hourly_csv": hourly_csv,
                            "annual_csv": annual_csv,
                            "bui_json": bui_json_path,
                        },
                        "elapsed_s": round(time.time() - t0, 3),
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "status": "error",
                        "combo": combo,
                        "combo_tag": combo_tag,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                        "elapsed_s": round(time.time() - t0, 3),
                    }
                )

    finally:
        if epw_path and os.path.exists(epw_path):
            try:
                os.remove(epw_path)
            except OSError:
                pass

    total_time = time.time() - start_all
    ok = [r for r in results if r["status"] == "success"]
    ko = [r for r in results if r["status"] == "error"]

    return {
        "status": "completed",
        "source": "archetype" if archetype else "custom",
        "building": {"category": category, "country": country, "name": name},
        "weather_source": weather_source,
        "u_values_requested": {"roof": u_roof, "wall": u_wall, "window": u_window, "slab": u_slab},
        "ecm_options": opts,
        "include_baseline": include_baseline,
        "output_dir": output_dir,
        "bui_dir": bui_dir if save_bui else None,
        "summary": {
            "total": len(results),
            "successful": len(ok),
            "failed": len(ko),
            "total_time_s": round(total_time, 3),
        },
        "results": results,
    }


@app.post("/ecm_application/run_single_save", tags=["Simulation"])
async def ecm_application_run_single_save(
    # Input selection
    archetype: bool = Query(True, description="If True, use an archetype."),
    category: Optional[str] = Query(None, description="Building category. Required when archetype=True."),
    country: Optional[str] = Query(None, description="Country. Required when archetype=True."),
    name: Optional[str] = Query(None, description="Archetype name. Required when archetype=True."),
    weather_source: str = Query("pvgis", description="Weather source: 'pvgis' or 'epw'."),
    epw_file: Optional[UploadFile] = File(None, description="EPW weather file (required when weather_source='epw')."),
    # Envelope options (single scenario: no combinations)
    ecm_options: Optional[str] = Query(
        None,
        description="Comma-separated subset of elements to apply in one single run, e.g. 'wall,window'. If omitted, inferred from provided u-values.",
    ),
    u_wall: Optional[float] = Query(None, description="New wall U-value."),
    u_roof: Optional[float] = Query(None, description="New roof U-value."),
    u_window: Optional[float] = Query(None, description="New window U-value."),
    u_slab: Optional[float] = Query(None, description="New slab-to-ground U-value."),
    include_baseline: bool = Query(False, description="If True, also run baseline."),
    # Heat pump option (integrated here, not in a separate endpoint)
    use_heat_pump: bool = Query(False, description="If True, apply heat pump option in integrated UNI/PV results."),
    heat_pump_cop: float = Query(3.2, description="Heat pump COP used in integrated UNI/PV results."),
    # PV option (integrated here, not in a separate endpoint)
    use_pv: bool = Query(False, description="If True, run integrated ISO52016+UNI11300+PV analysis."),
    pv_kwp: Optional[float] = Query(None, description="Installed PV power [kWp]. Required when use_pv=true."),
    pv_tilt_deg: float = Query(30.0, description="PV tilt [deg]."),
    pv_azimuth_deg: float = Query(0.0, description="PV azimuth [deg] (0=south in PVGIS convention used by pipeline)."),
    pv_use_pvgis: bool = Query(True, description="If True, use PVGIS hourly PV generation."),
    pv_pvgis_loss_percent: float = Query(14.0, description="PVGIS loss percentage."),
    pv_pvgis_year: Optional[int] = Query(None, description="Optional fixed PVGIS year."),
    annual_pv_yield_kwh_per_kwp: float = Query(1400.0, description="Fallback annual PV yield [kWh/kWp]."),
    pv_battery_params_json: Optional[str] = Form(None, description="Optional JSON object for PV battery parameters."),
    # UNI11300 optional overrides
    uni_input_unit: Optional[str] = Query(None, description="Override UNI input unit ('Wh' or 'kWh')."),
    uni_heating_params_json: Optional[str] = Form(None, description="Optional JSON object override for UNI heating_params."),
    uni_cooling_params_json: Optional[str] = Form(None, description="Optional JSON object override for UNI cooling_params."),
    return_hourly_building: bool = Query(False, description="If True, include ISO hourly building output in integrated UNI/PV payload."),
    # Saving controls
    output_dir: str = Query("results/ecm_api_single", description="Folder where CSV results are saved."),
    save_bui: bool = Query(True, description="If True, save BUI JSON variants on disk."),
    bui_dir: str = Query("building_examples_ecm_api_single", description="Folder where BUI JSON variants are saved."),
) -> Dict[str, Any]:
    """
    Sequential runner without ECM combinations.

    If more U-values are provided (e.g. wall + window), a single scenario is simulated with all of them together.
    Optionally, the same run can include UNI11300 + PV and heat pump analysis.
    """
    if not archetype:
        raise HTTPException(status_code=400, detail="This endpoint currently supports archetype=true only.")
    if not category or not country or not name:
        raise HTTPException(status_code=400, detail="With archetype=true you must provide 'category', 'country' and 'name'.")

    match = next(
        (b for b in BUILDING_ARCHETYPES if b.get("category") == category and b.get("country") == country and b.get("name") == name),
        None,
    )
    if match is None:
        raise HTTPException(status_code=404, detail=f"No archetype found for category='{category}', country='{country}', name='{name}'.")

    base_bui = match["bui"]
    building_name = match.get("name", name) or "building"
    base_uni_cfg = copy.deepcopy(
        match.get("uni11300")
        or match.get("systems_archetype")
        or UNI11300_SIMULATION_EXAMPLE
    )

    # Which envelope changes are active (single scenario only)
    if ecm_options:
        opts = [x.strip().lower() for x in ecm_options.split(",") if x.strip()]
    else:
        opts = []
        if u_wall is not None:
            opts.append("wall")
        if u_roof is not None:
            opts.append("roof")
        if u_window is not None:
            opts.append("window")
        if u_slab is not None:
            opts.append("slab")

    allowed = {"wall", "roof", "window", "slab"}
    bad = [o for o in opts if o not in allowed]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid ecm_options={bad}. Allowed: wall, roof, window, slab.")
    if "wall" in opts and u_wall is None:
        raise HTTPException(status_code=400, detail="For option 'wall' you must provide u_wall.")
    if "roof" in opts and u_roof is None:
        raise HTTPException(status_code=400, detail="For option 'roof' you must provide u_roof.")
    if "window" in opts and u_window is None:
        raise HTTPException(status_code=400, detail="For option 'window' you must provide u_window.")
    if "slab" in opts and u_slab is None:
        raise HTTPException(status_code=400, detail="For option 'slab' you must provide u_slab.")

    if use_heat_pump and heat_pump_cop <= 0:
        raise HTTPException(status_code=400, detail="heat_pump_cop must be > 0.")
    if use_pv and (pv_kwp is None or pv_kwp <= 0):
        raise HTTPException(status_code=400, detail="When use_pv=true you must provide pv_kwp > 0.")

    run_modified = bool(opts) or bool(use_heat_pump) or bool(use_pv)
    if not run_modified and not include_baseline:
        raise HTTPException(
            status_code=400,
            detail="Nothing to run: provide at least one U-value and/or use_pv/use_heat_pump, or set include_baseline=true.",
        )

    def _parse_json_form_obj(raw: Optional[str], field_name: str) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a valid JSON string.") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a JSON object.")
        return parsed

    battery_params_payload = _parse_json_form_obj(pv_battery_params_json, "pv_battery_params_json")
    uni_heating_override = _parse_json_form_obj(uni_heating_params_json, "uni_heating_params_json")
    uni_cooling_override = _parse_json_form_obj(uni_cooling_params_json, "uni_cooling_params_json")

    uni_cfg = copy.deepcopy(base_uni_cfg if isinstance(base_uni_cfg, dict) else {})
    if uni_input_unit is not None:
        uni_cfg["input_unit"] = uni_input_unit
    if uni_heating_override:
        current_heating = uni_cfg.get("heating_params", {})
        if not isinstance(current_heating, dict):
            current_heating = {}
        uni_cfg["heating_params"] = _deep_merge_dict(current_heating, uni_heating_override)
    if uni_cooling_override:
        current_cooling = uni_cfg.get("cooling_params", {})
        if not isinstance(current_cooling, dict):
            current_cooling = {}
        uni_cfg["cooling_params"] = _deep_merge_dict(current_cooling, uni_cooling_override)

    # Weather EPW handling
    epw_path: Optional[str] = None
    if weather_source == "epw":
        if epw_file is None:
            raise HTTPException(status_code=400, detail="With weather_source='epw' you must upload 'epw_file'.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epw") as tmp:
            tmp.write(await epw_file.read())
            epw_path = tmp.name
    elif weather_source != "pvgis":
        raise HTTPException(status_code=400, detail="weather_source must be 'pvgis' or 'epw'.")

    _ensure_dir(output_dir)
    if save_bui:
        _ensure_dir(bui_dir)

    start_all = time.time()
    results: List[Dict[str, Any]] = []

    scenario_specs: List[Dict[str, Any]] = []
    if include_baseline:
        scenario_specs.append(
            {
                "scenario_id": "baseline",
                "combo": [],
                "apply_u": False,
                "run_integrated": False,
                "use_pv": False,
                "use_heat_pump": False,
            }
        )
    if run_modified:
        combo = sorted(set(opts + (["heat_pump"] if use_heat_pump else []) + (["pv"] if use_pv else [])))
        scenario_specs.append(
            {
                "scenario_id": "single",
                "combo": combo,
                "apply_u": bool(opts),
                "run_integrated": bool(use_pv or use_heat_pump),
                "use_pv": bool(use_pv),
                "use_heat_pump": bool(use_heat_pump),
            }
        )

    try:
        for spec in scenario_specs:
            t0 = time.time()
            combo = spec["combo"]
            combo_tag = ",".join(combo) if combo else "BASELINE"
            try:
                bui_variant = (
                    apply_u_values_to_bui(
                        base_bui,
                        use_roof=("roof" in opts),
                        use_wall=("wall" in opts),
                        use_window=("window" in opts),
                        use_slab=("slab" in opts),
                        u_roof=u_roof,
                        u_wall=u_wall,
                        u_window=u_window,
                        u_slab=u_slab,
                    )
                    if spec["apply_u"]
                    else base_bui
                )

                bui_json_path = _save_bui_variant(bui_variant, combo, bui_dir) if save_bui else None

                if weather_source == "pvgis":

                    @retry_on_transient_error()
                    def _run_single_pvgis():
                        return pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                            bui_variant,
                            weather_source="pvgis",
                            sankey_graph=False,
                        )

                    hourly_sim, annual_results_df = _run_single_pvgis()
                else:
                    hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                        bui_variant,
                        weather_source="epw",
                        path_weather_file=epw_path,
                        sankey_graph=False,
                    )

                hourly_csv = _build_name_file(
                    building_name=building_name,
                    combo=combo,
                    weather_source=weather_source,
                    epw_path=epw_path,
                    base_dir=output_dir,
                )
                pd.DataFrame(hourly_sim).to_csv(hourly_csv, index=False)
                annual_csv = str(Path(hourly_csv).with_name(Path(hourly_csv).stem + "__annual.csv"))
                pd.DataFrame(annual_results_df).to_csv(annual_csv, index=False)

                integrated_results = None
                if spec["run_integrated"]:
                    if spec["use_pv"]:
                        pv_cfg: Dict[str, Any] = {
                            "pv_kwp": float(pv_kwp),
                            "tilt_deg": float(pv_tilt_deg),
                            "azimuth_deg": float(pv_azimuth_deg),
                            "use_pvgis": bool(pv_use_pvgis),
                            "pvgis_loss_percent": float(pv_pvgis_loss_percent),
                            "annual_pv_yield_kwh_per_kwp": float(annual_pv_yield_kwh_per_kwp),
                        }
                        if pv_pvgis_year is not None:
                            pv_cfg["pvgis_year"] = int(pv_pvgis_year)
                        if battery_params_payload is not None:
                            pv_cfg["battery_params"] = battery_params_payload

                        integrated_results = _run_iso52016_uni11300_pv_pipeline(
                            bui=bui_variant,
                            pv=pv_cfg,
                            uni_cfg=uni_cfg,
                            return_hourly_building=return_hourly_building,
                            hourly_sim=hourly_sim,
                            use_heat_pump=bool(spec["use_heat_pump"]),
                            heat_pump_cop=float(heat_pump_cop),
                        )
                    else:
                        uni_results = _compute_uni11300_full_from_hourly_df(
                            hourly_df=hourly_sim,
                            input_unit=uni_cfg.get("input_unit", "Wh"),
                            heating_params_payload=uni_cfg.get("heating_params") or {},
                            cooling_params_payload=uni_cfg.get("cooling_params") or {},
                        )
                        try:
                            fp_electric = float((uni_cfg.get("heating_params") or {}).get("fp_electric", 2.18))
                        except Exception:
                            fp_electric = 2.18
                        uni_results = _apply_heat_pump_to_uni11300_results(
                            uni11300_results=uni_results,
                            heat_pump_cop=float(heat_pump_cop),
                            fp_electric=fp_electric,
                        )
                        integrated_results = {
                            "inputs": {
                                "heat_pump": {"enabled": True, "cop": float(heat_pump_cop)},
                                "uni11300": {
                                    "input_unit": str(uni_cfg.get("input_unit", "Wh")),
                                    "heating_params_overridden": bool(uni_cfg.get("heating_params")),
                                    "cooling_params_overridden": bool(uni_cfg.get("cooling_params")),
                                },
                            },
                            "results": {"uni11300": uni_results},
                        }
                        if return_hourly_building:
                            integrated_results["results"]["hourly_building"] = dataframe_to_records_safe(hourly_sim)

                results.append(
                    {
                        "status": "success",
                        "scenario_id": spec["scenario_id"],
                        "combo": combo,
                        "combo_tag": combo_tag,
                        "files": {
                            "hourly_csv": hourly_csv,
                            "annual_csv": annual_csv,
                            "bui_json": bui_json_path,
                        },
                        "integrated_results": integrated_results,
                        "elapsed_s": round(time.time() - t0, 3),
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "status": "error",
                        "scenario_id": spec["scenario_id"],
                        "combo": combo,
                        "combo_tag": combo_tag,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "elapsed_s": round(time.time() - t0, 3),
                    }
                )
    finally:
        if epw_path and os.path.exists(epw_path):
            try:
                os.remove(epw_path)
            except OSError:
                pass

    total_time = time.time() - start_all
    ok = [r for r in results if r["status"] == "success"]
    ko = [r for r in results if r["status"] == "error"]

    return {
        "status": "completed",
        "source": "archetype",
        "building": {"category": category, "country": country, "name": name},
        "weather_source": weather_source,
        "u_values_requested": {"roof": u_roof, "wall": u_wall, "window": u_window, "slab": u_slab},
        "single_scenario_elements": opts,
        "include_baseline": include_baseline,
        "use_heat_pump": use_heat_pump,
        "heat_pump_cop": heat_pump_cop if use_heat_pump else None,
        "use_pv": use_pv,
        "output_dir": output_dir,
        "bui_dir": bui_dir if save_bui else None,
        "summary": {
            "total": len(results),
            "successful": len(ok),
            "failed": len(ko),
            "total_time_s": round(total_time, 3),
        },
        "results": results,
    }



# =============================================================================
# CO2 endpoints
# =============================================================================

@app.get("/emission-factors", tags=["Co2 Emissions"])
def get_emission_factors(country: str = "IT"):
    """
    Return emission factors (kgCO2eq/kWh) for a given country code.
    """
    if country not in EMISSION_FACTORS:
        raise HTTPException(status_code=404, detail=f"Country {country} not found")

    return {
        "country": country,
        "emission_factors_kg_co2eq_per_kwh": EMISSION_FACTORS[country],
        "sources": list(EMISSION_FACTORS[country].keys()),
    }


@app.post("/calculate", response_model=EmissionResult, tags=["Co2 Emissions"])
def calculate_single_scenario(scenario: ScenarioInput):
    """
    Compute CO2e emissions for a single scenario.
    """
    result = calculate_emissions(
        energy_source=scenario.energy_source,
        annual_consumption_kwh=scenario.annual_consumption_kwh,
        country=scenario.country,
    )

    return EmissionResult(
        name=scenario.name,
        energy_source=scenario.energy_source.value,
        annual_consumption_kwh=scenario.annual_consumption_kwh,
        emission_factor_kg_per_kwh=result["emission_factor"],
        annual_emissions_kg_co2eq=result["annual_emissions_kg"],
        annual_emissions_ton_co2eq=result["annual_emissions_ton"],
        equivalent_trees=result["equivalent_trees"],
        equivalent_km_car=result["equivalent_km_car"],
    )


@app.post("/compare", response_model=ComparisonResult, tags=["Co2 Emissions"])
def compare_scenarios(input_data: MultiScenarioInput):
    """
    Compare scenarios by emissions. The first scenario is treated as the baseline.
    """
    if len(input_data.scenarios) < 2:
        raise HTTPException(status_code=400, detail="At least 2 scenarios are required for comparison")

    results: List[EmissionResult] = []
    for scenario in input_data.scenarios:
        r = calculate_emissions(
            energy_source=scenario.energy_source,
            annual_consumption_kwh=scenario.annual_consumption_kwh,
            country=scenario.country,
        )
        results.append(
            EmissionResult(
                name=scenario.name,
                energy_source=scenario.energy_source.value,
                annual_consumption_kwh=scenario.annual_consumption_kwh,
                emission_factor_kg_per_kwh=r["emission_factor"],
                annual_emissions_kg_co2eq=r["annual_emissions_kg"],
                annual_emissions_ton_co2eq=r["annual_emissions_ton"],
                equivalent_trees=r["equivalent_trees"],
                equivalent_km_car=r["equivalent_km_car"],
            )
        )

    baseline = results[0]
    baseline_emissions = baseline.annual_emissions_kg_co2eq

    savings: List[SavingResult] = []
    for res in results[1:]:
        s = calculate_savings(baseline_emissions, res.annual_emissions_kg_co2eq)
        savings.append(
            SavingResult(
                scenario_name=res.name,
                absolute_kg_co2eq=s["absolute_kg_co2eq"],
                absolute_ton_co2eq=s["absolute_ton_co2eq"],
                percentage=s["percentage"],
            )
        )

    best_scenario = min(results[1:], key=lambda x: x.annual_emissions_kg_co2eq).name

    return ComparisonResult(
        baseline=baseline,
        scenarios=results[1:],
        best_scenario=best_scenario,
        savings=savings,
    )


@app.post("/calculate-intervention", tags=["Co2 Emissions"])
def calculate_intervention_impact(payload: InterventionInput):
    """
    Compute the impact of a retrofit intervention in terms of CO2 equivalent.
    """
    current = calculate_emissions(
        energy_source=payload.current_source,
        annual_consumption_kwh=payload.current_consumption_kwh,
        country=payload.country,
    )

    if payload.new_consumption_kwh is None:
        final_consumption = payload.current_consumption_kwh * (1.0 - payload.energy_reduction_percentage / 100.0)
    else:
        final_consumption = payload.new_consumption_kwh

    final_source = payload.new_source if payload.new_source is not None else payload.current_source
    future = calculate_emissions(
        energy_source=final_source,
        annual_consumption_kwh=final_consumption,
        country=payload.country,
    )

    saving = calculate_savings(
        baseline_emissions=current["annual_emissions_kg"],
        scenario_emissions=future["annual_emissions_kg"],
    )

    return {
        "intervention_summary": {
            "energy_reduction": f"{payload.energy_reduction_percentage}%",
            "source_change": f"{payload.current_source.value} → {final_source.value if isinstance(final_source, EnergySource) else str(final_source)}",
            "consumption_change": f"{payload.current_consumption_kwh:.0f} → {final_consumption:.0f} kWh/year",
        },
        "current_scenario": {
            "emissions_kg_co2eq": round(current["annual_emissions_kg"], 2),
            "emissions_ton_co2eq": round(current["annual_emissions_ton"], 3),
        },
        "future_scenario": {
            "emissions_kg_co2eq": round(future["annual_emissions_kg"], 2),
            "emissions_ton_co2eq": round(future["annual_emissions_ton"], 3),
        },
        "savings": saving,
        "environmental_impact": {
            "trees_saved": current["equivalent_trees"] - future["equivalent_trees"],
            "km_car_avoided": current["equivalent_km_car"] - future["equivalent_km_car"],
        },
    }

# =============================================================================
# PV + Heat Pump endpoint
# =============================================================================

# Import the PV+HP core function (NOT the router endpoint)
try:
    from relife_forecasting.routes.pv_hp_analysis import analyze_pv_hp_from_uni11300, BatteryParams
except Exception:
    from routes.pv_hp_analysis import analyze_pv_hp_from_uni11300, BatteryParams  # type: ignore


def _compute_uni11300_full_from_hourly_df(
    hourly_df: pd.DataFrame,
    input_unit: str = "Wh",
    heating_params_payload: Optional[Dict[str, Any]] = None,
    cooling_params_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute UNI/TS 11300 results from ISO52016 hourly df.

    Returns FULL UNI output:
      - hourly_results (records)
      - summary
    """
    if hourly_df is None or hourly_df.empty:
        raise HTTPException(status_code=400, detail="Hourly simulation data is empty.")

    input_unit = str(input_unit or "Wh").strip().lower()
    if input_unit not in {"wh", "kwh"}:
        raise HTTPException(status_code=400, detail="input_unit must be 'Wh' or 'kWh'.")
    scale_to_kwh = 0.001 if input_unit == "wh" else 1.0

    heat_series = None
    cool_series = None

    if "Q_H" in hourly_df.columns:
        heat_series = hourly_df["Q_H"].astype(float)
    elif "Q_HC" in hourly_df.columns:
        heat_series = hourly_df["Q_HC"].clip(lower=0.0).astype(float)

    if "Q_C" in hourly_df.columns:
        cool_series = hourly_df["Q_C"].astype(float)
    elif "Q_HC" in hourly_df.columns:
        cool_series = (-hourly_df["Q_HC"]).clip(lower=0.0).astype(float)

    if heat_series is None and cool_series is None:
        raise HTTPException(status_code=400, detail="Hourly data must include Q_H/Q_C (or Q_HC).")

    df_ideal = pd.DataFrame(index=hourly_df.index)
    if heat_series is not None:
        df_ideal["Q_ideal_heat_kWh"] = heat_series * scale_to_kwh
    if cool_series is not None:
        df_ideal["Q_ideal_cool_kWh"] = cool_series * scale_to_kwh

    heating_params_payload = heating_params_payload or {}
    cooling_params_payload = cooling_params_payload or {}

    if not isinstance(heating_params_payload, dict):
        raise HTTPException(status_code=400, detail="'heating_params' must be a JSON object.")
    if not isinstance(cooling_params_payload, dict):
        raise HTTPException(status_code=400, detail="'cooling_params' must be a JSON object.")

    try:
        heating_params = HeatingSystemParams(**heating_params_payload)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid heating_params: {exc}") from exc

    try:
        cooling_params = CoolingSystemParams(**cooling_params_payload)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid cooling_params: {exc}") from exc

    results_df = compute_primary_energy_from_hourly_ideal(
        df_hourly=df_ideal,
        heat_col="Q_ideal_heat_kWh",
        cool_col="Q_ideal_cool_kWh",
        heating_params=heating_params,
        cooling_params=cooling_params,
    )

    summary_fields = [
        "Q_ideal_heat_kWh",
        "Q_ideal_cool_kWh",
        "E_delivered_thermal_kWh",
        "E_delivered_electric_heat_kWh",
        "E_delivered_electric_cool_kWh",
        "E_delivered_electric_total_kWh",
        "EP_heat_total_kWh",
        "EP_cool_total_kWh",
        "EP_total_kWh",
    ]
    summary = {col: float(results_df[col].sum()) for col in summary_fields if col in results_df.columns}

    return {
        "input_unit": input_unit,
        "ideal_unit": "kWh",
        "n_hours": int(len(results_df)),
        "hourly_results": dataframe_to_records_safe(results_df),
        "summary": summary,
    }


def _apply_heat_pump_to_uni11300_results(
    uni11300_results: Dict[str, Any],
    heat_pump_cop: float,
    fp_electric: float = 2.18,
) -> Dict[str, Any]:
    """
    Post-process UNI11300 hourly results to represent electric heat pump heating.
    """
    if heat_pump_cop <= 0:
        raise HTTPException(status_code=400, detail="heat_pump_cop must be > 0.")

    hourly_records = uni11300_results.get("hourly_results")
    if not isinstance(hourly_records, list) or not hourly_records:
        raise HTTPException(status_code=400, detail="UNI11300 results must include a non-empty 'hourly_results' list.")

    df = pd.DataFrame(hourly_records)
    if df.empty:
        raise HTTPException(status_code=400, detail="UNI11300 hourly_results parsed empty.")

    q_ideal_heat = pd.to_numeric(df.get("Q_ideal_heat_kWh", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    e_aux_heat = pd.to_numeric(
        df.get("E_aux_total_heat_kWh", df.get("E_delivered_electric_heat_kWh", 0.0)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    e_electric_cool = pd.to_numeric(df.get("E_delivered_electric_cool_kWh", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    e_hp_heating = (q_ideal_heat / float(heat_pump_cop)).clip(lower=0.0)
    e_electric_heat_total = e_hp_heating + e_aux_heat

    df["E_hp_heating_kWh"] = e_hp_heating
    df["E_delivered_electric_heat_kWh"] = e_electric_heat_total
    df["E_delivered_electric_total_kWh"] = e_electric_heat_total + e_electric_cool
    df["EP_heat_total_kWh"] = df["E_delivered_electric_heat_kWh"] * float(fp_electric)

    ep_cool = pd.to_numeric(df.get("EP_cool_total_kWh", 0.0), errors="coerce").fillna(0.0)
    df["EP_total_kWh"] = df["EP_heat_total_kWh"] + ep_cool

    summary_fields = [
        "Q_ideal_heat_kWh",
        "Q_ideal_cool_kWh",
        "E_delivered_thermal_kWh",
        "E_delivered_electric_heat_kWh",
        "E_delivered_electric_cool_kWh",
        "E_delivered_electric_total_kWh",
        "EP_heat_total_kWh",
        "EP_cool_total_kWh",
        "EP_total_kWh",
        "E_hp_heating_kWh",
    ]
    summary = {
        col: float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
        for col in summary_fields
        if col in df.columns
    }
    summary["heat_pump_cop"] = float(heat_pump_cop)

    out = copy.deepcopy(uni11300_results)
    out["hourly_results"] = dataframe_to_records_safe(df)
    out["summary"] = summary
    out["heat_pump_applied"] = True
    out["heat_pump_cop"] = float(heat_pump_cop)
    return out


def _run_iso52016_uni11300_pv_pipeline(
    *,
    bui: Dict[str, Any],
    pv: Dict[str, Any],
    uni_cfg: Optional[Dict[str, Any]] = None,
    return_hourly_building: bool = False,
    hourly_sim: Optional[pd.DataFrame] = None,
    use_heat_pump: bool = False,
    heat_pump_cop: float = 3.2,
) -> Dict[str, Any]:
    """
    Shared one-shot pipeline:
      1) ISO 52016 (from provided hourly df or new PVGIS simulation)
      2) UNI/TS 11300
      3) PV + optional battery matching
    """
    if bui is None or not isinstance(bui, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid 'bui' (must be a JSON object).")
    if not isinstance(pv, dict):
        raise HTTPException(status_code=400, detail="'pv' must be a JSON object.")

    pv_kwp = pv.get("pv_kwp")
    if pv_kwp is None:
        raise HTTPException(status_code=400, detail="Missing pv.pv_kwp")

    latitude = pv.get("latitude", bui.get("building", {}).get("latitude"))
    longitude = pv.get("longitude", bui.get("building", {}).get("longitude"))
    if latitude is None or longitude is None:
        raise HTTPException(
            status_code=400,
            detail="Missing coordinates. Provide pv.latitude/pv.longitude or bui.building.latitude/longitude.",
        )

    tilt_deg = float(pv.get("tilt_deg", 30.0))
    azimuth_deg = float(pv.get("azimuth_deg", 0.0))
    use_pvgis = bool(pv.get("use_pvgis", True))
    pvgis_loss_percent = float(pv.get("pvgis_loss_percent", 14.0))
    pvgis_year = pv.get("pvgis_year", None)
    pvgis_year = int(pvgis_year) if pvgis_year is not None else None
    annual_yield = float(pv.get("annual_pv_yield_kwh_per_kwp", 1400.0))

    battery_params_obj = None
    if pv.get("battery_params") is not None:
        if not isinstance(pv["battery_params"], dict):
            raise HTTPException(status_code=400, detail="pv.battery_params must be a JSON object.")
        try:
            battery_params_obj = BatteryParams(**pv["battery_params"])
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid pv.battery_params: {exc}") from exc

    uni_cfg = uni_cfg or {}
    if not isinstance(uni_cfg, dict):
        raise HTTPException(status_code=400, detail="'uni11300' must be a JSON object.")
    uni_input_unit = uni_cfg.get("input_unit", "Wh")
    heating_params_payload = uni_cfg.get("heating_params") or {}
    cooling_params_payload = uni_cfg.get("cooling_params") or {}

    if hourly_sim is None:
        try:
            hourly_sim, _ = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                bui,
                weather_source="pvgis",
                sankey_graph=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ISO52016 simulation failed: {type(exc).__name__}: {exc}") from exc
    elif not isinstance(hourly_sim, pd.DataFrame):
        hourly_sim = pd.DataFrame(hourly_sim)

    uni11300_results = _compute_uni11300_full_from_hourly_df(
        hourly_df=hourly_sim,
        input_unit=uni_input_unit,
        heating_params_payload=heating_params_payload,
        cooling_params_payload=cooling_params_payload,
    )

    if use_heat_pump:
        try:
            fp_electric = float((heating_params_payload or {}).get("fp_electric", 2.18))
        except Exception:
            fp_electric = 2.18
        uni11300_results = _apply_heat_pump_to_uni11300_results(
            uni11300_results=uni11300_results,
            heat_pump_cop=float(heat_pump_cop),
            fp_electric=fp_electric,
        )

    pv_hp_results = analyze_pv_hp_from_uni11300(
        uni_payload={"hourly_results": uni11300_results["hourly_results"]},
        pv_kwp=float(pv_kwp),
        latitude=float(latitude),
        longitude=float(longitude),
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        use_pvgis=use_pvgis,
        annual_pv_yield_kwh_per_kwp=annual_yield,
        battery_params=battery_params_obj,
        pvgis_loss_percent=pvgis_loss_percent,
        pvgis_year=pvgis_year,
    )

    out = {
        "inputs": {
            "pv": {
                "pv_kwp": float(pv_kwp),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "tilt_deg": float(tilt_deg),
                "azimuth_deg": float(azimuth_deg),
                "use_pvgis": bool(use_pvgis),
                "pvgis_loss_percent": float(pvgis_loss_percent),
                "pvgis_year": pvgis_year,
                "annual_pv_yield_kwh_per_kwp": float(annual_yield),
                "has_battery": battery_params_obj is not None,
            },
            "uni11300": {
                "input_unit": str(uni_input_unit),
                "heating_params_overridden": bool(heating_params_payload),
                "cooling_params_overridden": bool(cooling_params_payload),
            },
            "heat_pump": {
                "enabled": bool(use_heat_pump),
                "cop": float(heat_pump_cop) if use_heat_pump else None,
            },
        },
        "results": {
            "uni11300": uni11300_results,
            "pv_hp": pv_hp_results,
        },
    }

    if return_hourly_building:
        out["results"]["hourly_building"] = dataframe_to_records_safe(hourly_sim)

    return out


@app.post("/run/iso52016-uni11300-pv", tags=["Simulation"])
def run_iso52016_uni11300_pv(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    One-shot pipeline:
      1) ISO 52016 (hourly Q_H/Q_C from BUI)
      2) UNI/TS 11300 (delivered electricity)
      3) PV + optional battery matching (PVGIS or simplified fallback)

    JSON body (minimal):
    {
      "bui": {...},
      "pv": {
        "pv_kwp": 10,
        "tilt_deg": 30,
        "azimuth_deg": 0,
        "use_pvgis": true,
        "pvgis_loss_percent": 14,
        "pvgis_year": 2019,
        "annual_pv_yield_kwh_per_kwp": 1400,
        "battery_params": {...}
      },
      "uni11300": { "input_unit": "Wh", "heating_params": {...}, "cooling_params": {...} },
      "return_hourly_building": false
    }
    """
    bui = payload.get("bui")
    pv = payload.get("pv") or {}
    uni_cfg = payload.get("uni11300") or {}
    return_hourly_building = bool(payload.get("return_hourly_building", False))
    use_heat_pump = bool(payload.get("use_heat_pump", False))
    heat_pump_cop = float(payload.get("heat_pump_cop", 3.2))

    return _run_iso52016_uni11300_pv_pipeline(
        bui=bui,
        pv=pv,
        uni_cfg=uni_cfg,
        return_hourly_building=return_hourly_building,
        hourly_sim=None,
        use_heat_pump=use_heat_pump,
        heat_pump_cop=heat_pump_cop,
    )
