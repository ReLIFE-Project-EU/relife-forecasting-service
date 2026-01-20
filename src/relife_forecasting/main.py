# main.py
from __future__ import annotations

from importlib.metadata import version
import copy
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pybuildingenergy as pybui
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from relife_forecasting.config.logging import configure_logging
from relife_forecasting.routes import health

# Prefer your package imports; keep a small fallback to reduce friction during refactors.
try:
    from relife_forecasting.building_examples import BUILDING_ARCHETYPES
except Exception:
    from building_examples import BUILDING_ARCHETYPES  # type: ignore

try:
    from relife_forecasting.routes.EPC_Greece_converter import U_VALUES_BY_CLASS, _norm_surface_name
except Exception:
    from routes.EPC_Greece_converter import U_VALUES_BY_CLASS, _norm_surface_name  # type: ignore

try:
    from relife_forecasting.routes.forecasting_service_functions import *
except Exception:
    from routes.forecasting_service_functions import *


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
        hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
            bui_checked,
            weather_source="pvgis",
        )

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
            )
    else:
        raise HTTPException(status_code=400, detail="weather_source must be either 'pvgis' or 'epw'.")

    # 4) ISO 15316 heating system simulation
    calc = pybui.HeatingSystemCalculator(system_checked)
    calc.load_csv_data(hourly_sim)
    df_system = calc.run_timeseries()

    # 5) Response
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
            # keep commented if you still want to reduce payload size:
            # "annual_building": dataframe_to_records_safe(annual_results_df),
            # "hourly_system": dataframe_to_records_safe(df_system),
        },
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
    payload: Optional[Dict[str, Any]] = Body(None, description="JSON body with 'bui' (and optionally 'system') when archetype=False."),
):
    """
    Update envelope U-values in the BUI according to an energy class (A/B/C/D).

    Behavior:
      - archetype=True: creates a NEW archetype and appends it to BUILDING_ARCHETYPES.
      - archetype=False: modifies a provided BUI and returns it (no persistence).
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
        if bui_json is None:
            raise HTTPException(status_code=400, detail="Incomplete JSON body: missing 'bui' key.")

        bui_internal = json_to_internal_bui(bui_json)

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
    }


# =============================================================================
# ECM endpoint (U-values scenario simulation)
# =============================================================================

@app.post("/ecm_application", tags=["Simulation"])
async def simulate_uvalues(
    archetype: bool = Query(True, description="If True, use an archetype; if False, use a custom BUI from JSON."),
    category: Optional[str] = Query(None, description="Building category. Required when archetype=True."),
    country: Optional[str] = Query(None, description="Country. Required when archetype=True."),
    name: Optional[str] = Query(None, description="Archetype name. Required when archetype=True."),
    weather_source: str = Query("pvgis", description="Weather source: 'pvgis' or 'epw'."),
    epw_file: Optional[UploadFile] = File(None, description="EPW weather file (required when weather_source='epw')."),
    bui_json: Optional[str] = Form(None, description="JSON string with the BUI (required when archetype=False)."),
    u_wall: Optional[float] = Query(None, description="New wall U-value (opaque vertical surfaces)."),
    u_roof: Optional[float] = Query(None, description="New roof U-value (opaque horizontal surface)."),
    u_window: Optional[float] = Query(None, description="New window U-value (transparent vertical surfaces)."),
):
    """
    Build all non-empty combinations of the requested U-value interventions and simulate each scenario.
    """
    # 1) Base BUI
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
    else:
        if bui_json is None:
            raise HTTPException(status_code=400, detail="With archetype=false you must send 'bui_json' as a form field.")
        try:
            base_bui = json.loads(bui_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="'bui_json' must be a valid JSON string.")

    scenarios_spec = build_uvalue_scenarios(u_roof=u_roof, u_wall=u_wall, u_window=u_window)
    if not scenarios_spec:
        raise HTTPException(status_code=400, detail="You must specify at least one of u_wall, u_roof, u_window.")

    # 2) Weather handling for EPW
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
        for spec in scenarios_spec:
            bui_variant = apply_u_values_to_bui(
                base_bui,
                use_roof=spec["use_roof"],
                use_wall=spec["use_wall"],
                use_window=spec["use_window"],
                u_roof=u_roof,
                u_wall=u_wall,
                u_window=u_window,
            )

            if weather_source == "pvgis":
                hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                    bui_variant,
                    weather_source="pvgis",
                )
            else:
                hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                    bui_variant,
                    weather_source="epw",
                    path_weather_file=epw_path,
                )

            scenario_results.append(
                {
                    "scenario_id": spec["id"],
                    "description": spec["label"],
                    "u_values": {"roof": u_roof, "wall": u_wall, "window": u_window},
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
        "u_values_requested": {"roof": u_roof, "wall": u_wall, "window": u_window},
        "n_scenarios": len(scenarios_spec),
        "scenarios": scenario_results,
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
