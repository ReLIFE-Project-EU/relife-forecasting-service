from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from relife_forecasting.models.forecasting import Project
# main.py
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile, File, Form, Query, Body
from fastapi.responses import HTMLResponse
import pybuildingenergy as pybui
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------
# Import example building archetypes
# ---------------------------------------
from building_examples import BUILDING_ARCHETYPES
from routes.EPC_Greece_converter import U_VALUES_BY_CLASS, _norm_surface_name



router = APIRouter(tags=["forecasting"])

# -------------------------------
# In-memory storage (demo only)
# -------------------------------

PROJECTS: Dict[str, Project] = {}



# ============================================================================
# Utility functions
# ============================================================================

def numpyfy(obj: Any) -> Any:
    """
    Optional utility: if you ever need to convert some list fields
    to np.array for specific keys, do it here.

    For now this is just a pass-through.
    """
    return obj


def dict_to_df_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a pandas DataFrame into a list of dictionaries
    (one dict per row), convenient for JSON responses.
    """
    return df.to_dict(orient="records")


def validate_bui_and_system(
    bui: Dict[str, Any],
    system: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate both the BUI (building input) and HVAC system configuration.

    Steps:
      1) If available, run pybui.check_heating_system_inputs(system)
         to check and normalize the HVAC inputs.
      2) Run sanitize_and_validate_BUI(bui, fix=True) to auto-fix
         some issues and get a 'fixed' BUI.
      3) Run sanitize_and_validate_BUI(bui_fixed, fix=False) to check
         for remaining issues without applying fixes.
      4) If any ERROR-level issues are found, raise HTTP 422.

    Returns a dict containing:
      - bui_fixed
      - bui_checked
      - system_checked
      - bui_report_fixed
      - bui_issues
      - system_messages
    """
    # 1) HVAC system check (if function exists in this pybuildingenergy version)
    system_messages: List[str] = []
    system_checked = system

    try:
        res_sys = pybui.check_heating_system_inputs(system)
        system_checked = res_sys["config"]
        system_messages.extend(res_sys.get("messages", []))
    except AttributeError:
        # pybuildingenergy version without check_heating_system_inputs
        system_messages.append(
            "⚠️ Function 'check_heating_system_inputs' is not available in the "
            "installed pybuildingenergy version: HVAC inputs are NOT validated automatically."
        )

    # 2) BUI: fix=True (sanitize and auto-fix some inputs)
    bui_fixed, report_fixed = pybui.sanitize_and_validate_BUI(bui, fix=True)

    # 3) BUI: only validation (no fixes)
    bui_checked, issues = pybui.sanitize_and_validate_BUI(bui_fixed, fix=False)
    errors = [e for e in issues if e["level"] == "ERROR"]

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


# ============================================================================
# JSON -> internal structures
# ============================================================================

def json_to_internal_bui(bui_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert JSON BUI representation into the internal format expected
    by pybuildingenergy.

    In particular:
      - For each 'adjacent_zones' entry, some fields are converted
        from list to np.array (dtype=object), since the library
        expects numpy arrays.
    """
    bui = copy.deepcopy(bui_json)

    for zone in bui.get("adjacent_zones", []):
        for key in [
            "area_facade_elements",
            "typology_elements",
            "transmittance_U_elements",
            "orientation_elements",
        ]:
            if key in zone and isinstance(zone[key], list):
                zone[key] = np.array(zone[key], dtype=object)

    return bui


def json_to_internal_system(system_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert JSON system representation into the internal format expected
    by pybuildingenergy.

    In particular:
      - 'gen_outdoor_temp_data' is converted from list[dict] to
        a pandas DataFrame, with a fixed index label.
    """
    system = copy.deepcopy(system_json)

    # gen_outdoor_temp_data: from list[dict] -> pd.DataFrame
    if "gen_outdoor_temp_data" in system and isinstance(system["gen_outdoor_temp_data"], list):
        df = pd.DataFrame(system["gen_outdoor_temp_data"])
        df.index = ["Generator curve"] * len(df)
        system["gen_outdoor_temp_data"] = df

    return system


# ============================================================================
# Worker for parallel simulation (single building)
# ============================================================================

def simulate_building_worker(building_name: str, bui: dict, system: dict) -> Dict[str, Any]:
    """
    Worker function for multiprocessing: simulate a single building.

    It performs:
      - ISO 52016 building needs calculation (hourly + annual)
      - ISO 15316 heating system simulation
      - basic energy KPIs (kWh, kWh/m²)

    Returns:
      A dict with:
        - name
        - status ("ok" or "error")
        - summary: small dict with KPIs
        - hourly: JSON-serializable time series for the building
        - annual: JSON-serializable annual results
        - system: JSON-serializable heating system time series
    """
    try:
        # --- ISO 52016 (building) ---
        hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
            bui,
            weather_source="pvgis",
        )

        # --- ISO 15316 (heating system) ---
        calc = pybui.HeatingSystemCalculator(system)
        calc.load_csv_data(hourly_sim)
        df_system = calc.run_timeseries()

        # --- Basic KPIs ---
        heating_kWh = hourly_sim.loc[hourly_sim["Q_HC"] > 0, "Q_HC"].sum() / 1000
        cooling_kWh = -hourly_sim.loc[hourly_sim["Q_HC"] < 0, "Q_HC"].sum() / 1000
        area = bui["building"].get("net_floor_area", None)

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
            "hourly": to_jsonable(hourly_sim),
            "annual": to_jsonable(annual_results_df),
            "system": to_jsonable(df_system),
        }

    except Exception as e:
        return {
            "name": building_name,
            "status": "error",
            "error": str(e),
        }


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/templates", tags=["Templates"])
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
    # --- ARCHETYPE CASE ---
    if archetype:
        if not category or not country or not name:
            raise HTTPException(
                status_code=400,
                detail=(
                    "With archetype=true you must provide 'category', 'country' and 'name' "
                    "as query parameters."
                ),
            )

        match = next(
            (
                b
                for b in BUILDING_ARCHETYPES
                if b["category"] == category
                and b["country"] == country
                and b["name"] == name
            ),
            None,
        )

        if match is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No archetype found for "
                    f"category='{category}', country='{country}', name='{name}'."
                ),
            )

        return {
            "source": "archetype",
            "name": match["name"],
            "category": match["category"],
            "country": match["country"],
            "bui": to_jsonable(match["bui"]),
            "system": to_jsonable(match["system"]),
        }

    # --- CUSTOM CASE ---
    if payload is None:
        raise HTTPException(
            status_code=400,
            detail="With archetype=false you must send a JSON body containing 'bui' and 'system'.",
        )

    bui_json = payload.get("bui")
    system_json = payload.get("system")

    if bui_json is None or system_json is None:
        raise HTTPException(
            status_code=400,
            detail="Incomplete JSON body: both 'bui' and 'system' are required.",
        )

    return {
        "source": "custom",
        "name": name,
        "category": category,
        "country": country,
        "bui": bui_json,
        "system": system_json,
    }


@router.get("/templates/available", tags=["Templates"])
def list_available_archetypes():
    """
    List available archetypes (metadata only, without full BUI/HVAC content).
    """
    return [
        {
            "name": b["name"],
            "category": b["category"],
            "country": b["country"],
        }
        for b in BUILDING_ARCHETYPES
    ]


@router.post("/validate", tags=["Validation"])
def validate_model(
    archetype: bool = Query(
        True,
        description="If True, validate an archetype (category+country+name). If False, validate a custom model from the body.",
    ),
    category: Optional[str] = Query(
        None,
        description="Building category. Required when archetype=True.",
    ),
    country: Optional[str] = Query(
        None,
        description="Country. Required when archetype=True.",
    ),
    name: Optional[str] = Query(
        None,
        description="Archetype name. Required when archetype=True.",
    ),
    payload: Optional[Dict[str, Any]] = Body(
        None,
        description="JSON body with 'bui' and 'system' when archetype=False.",
    ),
):
    """
    Validate BUI and HVAC system for either:
      - an archetype (when archetype=True), or
      - a custom input model provided in the request body.
    """
    # -------- 1) Archetype --------
    if archetype:
        if not category or not country or not name:
            raise HTTPException(
                status_code=400,
                detail=(
                    "With archetype=true you must provide 'category', 'country' and 'name' "
                    "as query parameters."
                ),
            )

        match = next(
            (
                b
                for b in BUILDING_ARCHETYPES
                if b["category"] == category
                and b["country"] == country
                and b["name"] == name
            ),
            None,
        )

        if match is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No archetype found for "
                    f"category='{category}', country='{country}', name='{name}'."
                ),
            )

        bui_internal = match["bui"]
        system_internal = match["system"]

    # -------- 2) Custom --------
    else:
        if payload is None:
            raise HTTPException(
                status_code=400,
                detail="With archetype=false you must send a JSON body containing 'bui' and 'system'.",
            )

        bui_json = payload.get("bui")
        system_json = payload.get("system")

        if bui_json is None or system_json is None:
            raise HTTPException(
                status_code=400,
                detail="Incomplete JSON body: both 'bui' and 'system' are required.",
            )

        bui_internal = json_to_internal_bui(bui_json)
        system_internal = json_to_internal_system(system_json)

    # -------- 3) Common validation --------
    result = validate_bui_and_system(bui_internal, system_internal)

    return {
        "source": "archetype" if archetype else "custom",
        "name": name,
        "category": category,
        "country": country,
        "bui_fixed": to_jsonable(result["bui_fixed"]),
        "bui_checked": to_jsonable(result["bui_checked"]),
        "system_checked": to_jsonable(result["system_checked"]),
        "bui_report_fixed": to_jsonable(result["bui_report_fixed"]),
        "bui_issues": result["bui_issues"],
        "system_messages": result["system_messages"],
    }


@router.post("/simulate", tags=["Simulation"])
async def simulate_building(
    archetype: bool = Query(
        True,
        description="If True, simulate an archetype. If False, use custom BUI+system.",
    ),
    category: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    weather_source: str = Query(
        "pvgis", description="Weather source: pvgis or epw"
    ),
    epw_file: Optional[UploadFile] = File(
        None,
        description="EPW weather file (required if weather_source='epw')"
    ),
    payload: Optional[Dict[str, Any]] = Body(
        None,
        description="Required when archetype=False: JSON with 'bui' and 'system'",
    ),
):
    """
    Simulate a building (ISO 52016 + ISO 15316).

    - Uses PVGIS or EPW weather file.
    - EPW file must be uploaded via multipart/form-data when weather_source='epw'.
    """
    # ---------------- 1) Select BUI/SYSTEM ----------------
    if archetype:
        if not category or not country or not name:
            raise HTTPException(
                400,
                "For archetype=True you must specify category, country and name.",
            )

        match = next(
            (b for b in BUILDING_ARCHETYPES
             if b["category"] == category
             and b["country"] == country
             and b["name"] == name),
            None,
        )

        if match is None:
            raise HTTPException(
                404,
                f"No archetype found for {category}/{country}/{name}",
            )

        bui_internal = match["bui"]
        system_internal = match["system"]

    else:
        if payload is None or "bui" not in payload or "system" not in payload:
            raise HTTPException(
                400,
                "With archetype=False you must send a JSON body with 'bui' and 'system'."
            )

        bui_internal = json_to_internal_bui(payload["bui"])
        system_internal = json_to_internal_system(payload["system"])

    # ---------------- 2) Validate ----------------
    validated = validate_bui_and_system(bui_internal, system_internal)
    bui_checked = validated["bui_checked"]
    system_checked = validated["system_checked"]

    # ---------------- 3) Weather ----------------
    if weather_source == "pvgis":
        hourly_sim, annual_results_df = (
            pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                bui_checked,
                weather_source="pvgis",
            )
        )

    elif weather_source == "epw":
        if epw_file is None:
            raise HTTPException(
                400,
                "When weather_source='epw', you must upload an EPW file using epw_file."
            )

        # Save uploaded EPW to a temporary folder
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            epw_path = os.path.join(tmpdir, epw_file.filename)
            with open(epw_path, "wb") as f:
                f.write(await epw_file.read())

            # Run simulation
            hourly_sim, annual_results_df = (
                pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                    bui_checked,
                    weather_source="epw",
                    weather_file=epw_path,   # same key used by simulate_epw
                )
            )

    else:
        raise HTTPException(400, "weather_source must be 'pvgis' or 'epw'.")

    # ---------------- 4) Heating System ----------------
    calc = pybui.HeatingSystemCalculator(system_checked)
    calc.load_csv_data(hourly_sim)
    df_heating_system = calc.run_timeseries()

    # ---------------- 5) Return ----------------
    return {
        "weather_source": weather_source,
        "archetype": archetype,
        "category": category,
        "country": country,
        "name": name,
        "validation": {
            "bui_issues": validated["bui_issues"],
            "system_messages": validated["system_messages"],
        },
        "results": {
            "hourly_building": to_jsonable(hourly_sim),
            "annual_building": to_jsonable(annual_results_df),
            "hourly_system": to_jsonable(df_heating_system),
        },
    }




@router.post("/report", response_class=HTMLResponse, tags=["Reports"])
def generate_report(payload: Dict[str, Any]):
    """
    Generate an HTML report with hourly/monthly/annual statistical analysis
    of energy needs.

    Input JSON must contain:
      - hourly_sim: list of records (rows) representing the hourly DataFrame
      - building_area: float (m²)
    """
    hourly_records = payload.get("hourly_sim")
    building_area = payload.get("building_area")

    if hourly_records is None or building_area is None:
        raise HTTPException(
            status_code=400,
            detail="Request JSON must contain 'hourly_sim' and 'building_area'.",
        )

    # Rebuild DataFrame from JSON records
    hourly_sim = pd.DataFrame(hourly_records)

    # Temporary directory to store the HTML report
    with tempfile.TemporaryDirectory() as tmpdir:
        name_report = "energy_report"

        # Build graph/report object
        report_obj = pybui.Graphs_and_report(
            df=hourly_sim,
            season="heating_cooling",
            building_area=building_area,
        )

        # This writes the HTML file to disk
        report_obj.bui_analysis_page(
            folder_directory=tmpdir,
            name_file=name_report,
        )

        html_path = os.path.join(tmpdir, f"{name_report}.html")
        if not os.path.exists(html_path):
            raise HTTPException(
                status_code=500,
                detail="HTML report not found. Check the 'bui_analysis_page' implementation.",
            )

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

    # Return the HTML content directly in the response
    return HTMLResponse(content=html_content)


@router.post("/simulate_epw", tags=["Simulation"])
async def simulate_with_epw(
    epw_file: UploadFile = File(...),
    bui_json: str = Form(...),
    system_json: str = Form(...),
):
    """
    Simulate a building using an EPW weather file.

    Parameters (multipart/form-data):
      - epw_file: uploaded EPW file
      - bui_json: string containing BUI in JSON format
      - system_json: string containing HVAC system in JSON format

    Example (curl):

      curl -X POST http://localhost:8000/simulate_epw \\
           -F "epw_file=@weather.epw" \\
           -F 'bui_json={...}' \\
           -F 'system_json={...}'
    """
    try:
        bui = json.loads(bui_json)
        system = json.loads(system_json)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="bui_json and system_json must be valid JSON strings.",
        )

    bui = numpyfy(bui)
    system = numpyfy(system)

    validated = validate_bui_and_system(bui, system)
    bui_checked = validated["bui_checked"]
    system_checked = validated["system_checked"]

    # Save EPW in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        epw_path = os.path.join(tmpdir, epw_file.filename)
        with open(epw_path, "wb") as f:
            f.write(await epw_file.read())

        # Adapt these keyword arguments to match your pybuildingenergy API
        hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
            bui_checked,
            weather_source="epw",
            weather_file=epw_path,   # Adapt if your function expects a different argument name
        )

        calc = pybui.HeatingSystemCalculator(system_checked)
        calc.load_csv_data(hourly_sim)
        df_heating_system = calc.run_timeseries()

    return {
        "status": "ok",
        "weather_source": "epw",
        "hourly_needs": dict_to_df_records(hourly_sim),
        "annual_needs": dict_to_df_records(annual_results_df),
        "heating_system": dict_to_df_records(df_heating_system),
    }


@router.post("/bui/update_u_values", tags=["BUI"])
def update_u_values(
    energy_class: str = Query(
        ...,
        description="Envelope performance class: A, B, C or D.",
        regex="^[ABCD]$",
    ),
    archetype: bool = Query(
        True,
        description=(
            "If True, load the BUI from an archetype (category+country+name) "
            "and create a NEW modified archetype. "
            "If False, use a custom BUI from the request body (not stored)."
        ),
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
        description="Name of the source archetype. Required when archetype=True.",
    ),
    new_name: Optional[str] = Query(
        None,
        description=(
            "Name of the NEW archetype to be created. "
            "If not provided, a default name will be generated."
        ),
    ),
    payload: Optional[Dict[str, Any]] = Body(
        None,
        description="JSON body with 'bui' (and optionally 'system') when archetype=False.",
    ),
):
    """
    Update envelope U-values in the BUI according to an energy class (A/B/C/D).

    Behavior:
      - If archetype=True:
          * Load BUI/system from BUILDING_ARCHETYPES (category+country+name).
          * Apply new U-values based on 'energy_class'.
          * Create a NEW archetype (with 'new_name' or auto-generated name)
            and append it to BUILDING_ARCHETYPES.
      - If archetype=False:
          * Read a custom BUI from the request body and modify it in-place
            (no archetype is created).
    """
    # --- 1) Load base BUI (and system) ---

    base_category = category
    base_country = country
    base_name = name
    base_system = None

    if archetype:
        if not category or not country or not name:
            raise HTTPException(
                status_code=400,
                detail=(
                    "With archetype=true you must provide 'category', 'country' and 'name' "
                    "as query parameters."
                ),
            )

        match = next(
            (
                b
                for b in BUILDING_ARCHETYPES
                if b["category"] == category
                and b["country"] == country
                and b["name"] == name
            ),
            None,
        )

        if match is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No archetype found for "
                    f"category='{category}', country='{country}', name='{name}'."
                ),
            )

        # Copy BUI and system so we don't modify the original archetype
        bui_internal = copy.deepcopy(match["bui"])
        base_system = copy.deepcopy(match["system"])

    else:
        # Custom case: only BUI is modified, no archetype is created
        if payload is None:
            raise HTTPException(
                status_code=400,
                detail="With archetype=false you must send a JSON body with at least 'bui'.",
            )

        bui_json = payload.get("bui")
        if bui_json is None:
            raise HTTPException(
                status_code=400,
                detail="Incomplete JSON body: missing 'bui' key.",
            )

        bui_internal = json_to_internal_bui(bui_json)

    # --- 2) Update U-values according to the selected energy class ---

    class_map = U_VALUES_BY_CLASS.get(energy_class)
    if class_map is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid energy class '{energy_class}'. Use A, B, C or D.",
        )

    surfaces = bui_internal.get("building_surface", [])

    for surf in surfaces:
        name_surf = surf.get("name")
        if not name_surf:
            continue

        key = _norm_surface_name(name_surf)
        new_u = class_map.get(key)

        if new_u is not None:
            surf["u_value"] = float(new_u)

    # --- 3) If working on an archetype, create and store a NEW archetype ---

    created_archetype_name = None

    if archetype:
        # Use provided new_name or generate a default one
        if new_name:
            created_archetype_name = new_name
        else:
            created_archetype_name = f"{base_name}_class_{energy_class}"

        new_archetype = {
            "name": created_archetype_name,
            "category": base_category,
            "country": base_country,
            "bui": bui_internal,
            "system": base_system,
        }

        # Append the new archetype to the global list
        BUILDING_ARCHETYPES.append(new_archetype)

    # --- 4) Response ---

    return {
        "source": "archetype" if archetype else "custom",
        "base_name": base_name,
        "base_category": base_category,
        "base_country": base_country,
        "energy_class": energy_class,
        "new_archetype_name": created_archetype_name,
        "bui": to_jsonable(bui_internal),
    }


@router.post("/simulate/batch", tags=["Batch Simulation"])
def simulate_batch(payload: Dict[str, Any]):
    """
    Batch-simulate multiple buildings in parallel (multiprocessing).

    Two modes are supported:

      mode = "archetype":
        {
          "mode": "archetype",
          "category": "Single Family House",
          "countries": ["Italy", "Greece"],
          "names": ["SFH_Italy_default", "SFH_Greece_default"]
        }

        - category: required
        - countries: list of countries to include (required)
        - names: list of archetype names to include (required)

      mode = "custom":
        {
          "mode": "custom",
          "buildings": [
            { "name": "B1", "bui": {...}, "system": {...} },
            { "name": "B2", "bui": {...}, "system": {...} }
          ]
        }

        Each building must have:
          - name
          - bui: JSON BUI
          - system: JSON system
    """
    mode = payload.get("mode")
    buildings_to_simulate: List[Dict[str, Any]] = []  # list of dicts: {"name", "bui", "system"}

    # =================== ARCHETYPE MODE ===================
    if mode == "archetype":
        category = payload.get("category")
        countries = payload.get("countries") or []
        names = payload.get("names") or []

        if not category or not countries or not names:
            raise HTTPException(
                status_code=400,
                detail=(
                    "For mode='archetype' you must provide 'category', "
                    "'countries' (list) and 'names' (list)."
                ),
            )

        # Filter BUILDING_ARCHETYPES by category, country in countries, name in names
        for arch in BUILDING_ARCHETYPES:
            if (
                arch.get("category") == category
                and arch.get("country") in countries
                and arch.get("name") in names
            ):
                buildings_to_simulate.append(
                    {
                        "name": arch["name"],
                        "bui": arch["bui"],
                        "system": arch["system"],
                        "category": arch.get("category"),
                        "country": arch.get("country"),
                    }
                )

        if not buildings_to_simulate:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No archetypes found for the given parameters: "
                    f"category='{category}', countries={countries}, names={names}."
                ),
            )

    # =================== CUSTOM MODE ======================
    elif mode == "custom":
        buildings = payload.get("buildings")
        if not buildings:
            raise HTTPException(
                status_code=400,
                detail="For mode='custom' you must provide 'buildings': [ {name,bui,system} ].",
            )

        for b in buildings:
            name = b.get("name")
            bui_json = b.get("bui")
            system_json = b.get("system")

            if not name or bui_json is None or system_json is None:
                raise HTTPException(
                    status_code=400,
                    detail="Each custom building must have 'name', 'bui' and 'system'.",
                )

            buildings_to_simulate.append(
                {
                    "name": name,
                    "bui": json_to_internal_bui(bui_json),
                    "system": json_to_internal_system(system_json),
                    "category": b.get("category"),
                    "country": b.get("country"),
                }
            )

    else:
        raise HTTPException(
            status_code=400,
            detail="Field 'mode' must be either 'archetype' or 'custom'.",
        )

    # =================== PARALLEL SIMULATION ======================

    results: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    with ProcessPoolExecutor() as executor:
        futures = []
        for b in buildings_to_simulate:
            futures.append(
                executor.submit(
                    simulate_building_worker,
                    b["name"],
                    b["bui"],
                    b["system"],
                )
            )

        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            if res.get("status") == "ok":
                summaries.append(res["summary"])

    # Summary DataFrame -> JSON
    summary_df = pd.DataFrame(summaries) if summaries else pd.DataFrame()

    return {
        "status": "completed",
        "mode": mode,
        "n_buildings": len(results),
        "results": results,  # detailed results per building
        "summary": summary_df.to_dict(orient="records"),  # compact summary table
    }