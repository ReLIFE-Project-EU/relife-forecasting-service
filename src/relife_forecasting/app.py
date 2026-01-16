from importlib.metadata import version

import copy
import itertools
import json
import math
import os
import tempfile
import pybuildingenergy as pybui
import numpy as np
import pandas as pd

from fastapi import FastAPI, APIRouter, File, HTTPException, UploadFile, Form, Query, Body
from fastapi.responses import HTMLResponse
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Any, Dict, List, Optional

from relife_forecasting.config.logging import configure_logging
from relife_forecasting.models.forecasting import Project
# from relife_forecasting.routes import auth, examples, forecasting, health
# from relife_forecasting.routes import forecasting
from relife_forecasting.routes import health


# ---------------------------------------
# Import example building archetypes
# ---------------------------------------
from relife_forecasting.building_examples import BUILDING_ARCHETYPES
from relife_forecasting.routes.EPC_Greece_converter import U_VALUES_BY_CLASS, _norm_surface_name

# Dynamically determine the package name
package_name = __name__.split(".")[0]

# Get version dynamically
package_dist_name = package_name.replace("_", "-")

try:
    __version__ = version(package_dist_name)
except ImportError:
    __version__ = "development"

configure_logging()

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
# app.include_router(auth.router)
# app.include_router(examples.router)
# app.include_router(forecasting.router)

router = APIRouter(tags=["forecasting"])

# -------------------------------
# In-memory storage (demo only)
# -------------------------------

PROJECTS: Dict[str, Project] = {}



# ============================================================================
# Utility functions
# ============================================================================
import pandas as pd
import numpy as np

from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(obj: Any):
    # tipi primitivi: già JSON-serializzabili
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # dizionari
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}

    # liste / tuple / set → lista JSON-safe
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # scalari numpy (np.float64, np.int64, ecc.)
    if isinstance(obj, np.generic):
        return obj.item()

    # pandas Series / Index
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()

    # pandas DataFrame (se mai ti capita nel BUI/system)
    if isinstance(obj, pd.DataFrame):
        # ad es. colonne → liste
        return {col: to_jsonable(obj[col].values) for col in obj.columns}

    # fallback: ultimo tentativo con .tolist(), poi stringa
    try:
        return obj.tolist()
    except AttributeError:
        return str(obj)

def clean_and_jsonable(obj: Any):
    """
    Converte oggetti complessi (numpy, pandas, ecc.) in strutture JSON-safe.
    - Mantiene stringhe e tipi primitivi così come sono
    - Converte array/Series/DataFrame in liste/dict
    - Gestisce correttamente array numerici e non numerici
    """
    # ---- tipi primitivi: già JSON-safe ----
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # ---- dict: ricorsione su valori ----
    if isinstance(obj, dict):
        return {k: clean_and_jsonable(v) for k, v in obj.items()}

    # ---- liste / tuple / set ----
    if isinstance(obj, (list, tuple, set)):
        return [clean_and_jsonable(x) for x in obj]

    # ---- numpy array ----
    if isinstance(obj, np.ndarray):
        # Se è chiaramente numerico, converto a float
        if np.issubdtype(obj.dtype, np.number):
            return obj.astype(float).tolist()
        else:
            # es. ['OP', 'GR', ...] oppure dtype=object misto
            return [clean_and_jsonable(x) for x in obj.tolist()]

    # ---- scalari numpy (np.float64, np.int64, ecc.) ----
    if isinstance(obj, np.generic):
        # se numerico → float, altrimenti valore "puro"
        if np.issubdtype(type(obj), np.number):
            return float(obj)
        return obj.item()

    # ---- pandas Series / Index ----
    if isinstance(obj, (pd.Series, pd.Index)):
        return [clean_and_jsonable(x) for x in obj.tolist()]

    # ---- pandas DataFrame ----
    if isinstance(obj, pd.DataFrame):
        # esempio: {col: [valori ...]} – JSON-friendly
        return {
            col: clean_and_jsonable(obj[col].values)
            for col in obj.columns
        }

    # ---- fallback: ultimo tentativo con .tolist(), poi stringa ----
    try:
        return obj.tolist()
    except AttributeError:
        return str(obj)


# def to_jsonable(obj):
#     """Converte DataFrame, Series, numpy array e tipi non JSON-safe 
#     in strutture JSON-serializzabili."""
    
#     if isinstance(obj, pd.DataFrame):
#         return obj.to_dict(orient="records")
    
#     if isinstance(obj, pd.Series):
#         return obj.to_dict()

#     if isinstance(obj, (np.ndarray, list, tuple)):
#         return obj.tolist()

#     if isinstance(obj, (np.integer, np.floating)):
#         return obj.item()

#     if isinstance(obj, dict):
#         return {k: to_jsonable(v) for k, v in obj.items()}

#     # fallback: lasciamo che FastAPI gestisca i tipi base
#     return obj


# def clean_and_jsonable(obj):
#     """Converte DataFrame/Series/array e sostituisce NaN/inf con None per JSON."""
    
#     # Pandas DataFrame
#     if isinstance(obj, pd.DataFrame):
#         df = obj.replace([np.inf, -np.inf], np.nan)
#         # sostituisce NaN con None
#         return df.where(pd.notnull(df), None).to_dict(orient="records")
    
#     # Pandas Series
#     if isinstance(obj, pd.Series):
#         s = obj.replace([np.inf, -np.inf], np.nan)
#         return s.where(pd.notnull(s), None).to_dict()

#     # numpy array
#     if isinstance(obj, np.ndarray):
#         arr = obj.astype(float)
#         arr[~np.isfinite(arr)] = np.nan
#         # convertiamo a lista, e sostituiamo NaN con None
#         return [
#             (None if (isinstance(x, float) and math.isnan(x)) else x)
#             for x in arr.tolist()
#         ]

#     # liste / tuple
#     if isinstance(obj, (list, tuple)):
#         return [clean_and_jsonable(x) for x in obj]

#     # dict
#     if isinstance(obj, dict):
#         return {k: clean_and_jsonable(v) for k, v in obj.items()}

#     # numpy scalari
#     if isinstance(obj, (np.floating, np.integer)):
#         v = obj.item()
#         if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
#             return None
#         return v

#     # float normali
#     if isinstance(obj, float):
#         if math.isnan(obj) or math.isinf(obj):
#             return None
#         return obj

#     # tutto il resto lo lasciamo così
#     return obj



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
            "hourly": clean_and_jsonable(hourly_sim),
            "annual": clean_and_jsonable(annual_results_df),
            "system": clean_and_jsonable(df_system),
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


@app.get("/building/available", tags=["Building and systems"])
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


@app.post("/validate", tags=["Validation"])
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
        "bui_fixed": clean_and_jsonable(result["bui_fixed"]),
        "bui_checked": clean_and_jsonable(result["bui_checked"]),
        "system_checked": clean_and_jsonable(result["system_checked"]),
        "bui_report_fixed": clean_and_jsonable(result["bui_report_fixed"]),
        "bui_issues": result["bui_issues"],
        "system_messages": result["system_messages"],
    }



@app.post("/simulate", tags=["Simulation"])
async def simulate_building(
    archetype: bool = Query(
        True,
        description="If True, simulate an archetype (category+country+name). If False, use custom BUI/SYSTEM.",
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
    weather_source: str = Query(
        "pvgis",
        description="Weather source: 'pvgis' or 'epw'.",
    ),
    # 👇 EPW file upload (solo se weather_source='epw')
    epw_file: Optional[UploadFile] = File(
        None,
        description="EPW weather file (required when weather_source='epw').",
    ),
    # 👇 usate solo se archetype=False
    bui_json: Optional[str] = Form(
        None,
        description="JSON string with the BUI data (required when archetype=False).",
    ),
    system_json: Optional[str] = Form(
        None,
        description="JSON string with the SYSTEM data (required when archetype=False).",
    ),
):
    """
    Simulate a single building (ISO 52016 + ISO 15316) using either:
      - an archetype (archetype=True), or
      - a custom BUI/system configuration (archetype=False).

    Weather:
      - weather_source='pvgis' → dati da PVGIS
      - weather_source='epw'   → si carica l'EPW via epw_file (UploadFile)
    """

    # -------- 1) Select BUI/System --------
    if archetype:
      # usa gli archetipi predefiniti
      if not category or not country or not name:
          raise HTTPException(
              status_code=400,
              detail="With archetype=true you must provide 'category', 'country' and 'name'.",
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

    else:
      # custom BUI/SYSTEM via JSON in form-data
      if bui_json is None or system_json is None:
          raise HTTPException(
              status_code=400,
              detail="With archetype=false you must send 'bui_json' and 'system_json' as form fields.",
          )

      try:
          bui_raw = json.loads(bui_json)
          system_raw = json.loads(system_json)
      except json.JSONDecodeError:
          raise HTTPException(
              status_code=400,
              detail="bui_json and system_json must be valid JSON strings.",
          )

      bui_internal = json_to_internal_bui(bui_raw)
      system_internal = json_to_internal_system(system_raw)

    # -------- 2) Validate BUI/System --------
    result = validate_bui_and_system(bui_internal, system_internal)
    bui_checked = result["bui_checked"]
    system_checked = result["system_checked"]

    # -------- 3) Weather configuration + ISO 52016 --------
    if weather_source == "pvgis":
        # nessun file, uso PVGIS
        hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
            bui_checked,
            weather_source="pvgis",
        )

    elif weather_source == "epw":
        if epw_file is None:
            raise HTTPException(
                status_code=400,
                detail="With weather_source='epw' you must upload 'epw_file'.",
            )

        # Salviamo il file caricato in una cartella temporanea
        with tempfile.TemporaryDirectory() as tmpdir:
            epw_path = os.path.join(tmpdir, epw_file.filename)

            # scriviamo il contenuto dell’UploadFile su disco
            contents = await epw_file.read()
            with open(epw_path, "wb") as f:
                f.write(contents)

            # 👈 QUI: usiamo il nome del parametro corretto
            hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                bui_checked,
                weather_source="epw",
                path_weather_file=epw_path,  # ✅ parametro giusto
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="weather_source must be either 'pvgis' or 'epw'.",
        )

    # -------- 4) ISO 15316 heating system simulation --------
    calc = pybui.HeatingSystemCalculator(system_checked)
    calc.load_csv_data(hourly_sim)  # expects columns like Q_HC / T_op / T_ext or aliases
    df_system = calc.run_timeseries()

    # -------- 5) Response --------
    def df_to_json_safe(df: pd.DataFrame):
        # 1. Se l'indice è Datetime, lo converte in colonna 'timestamp'
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={'index': 'timestamp'})
        else:
            # Altrimenti resetta l'indice e aggiunge un numero di riga
            df = df.reset_index(drop=True)
        
        # 2. Pulisce i valori problematici
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 3. Converte NaN in None (JSON null)
        df = df.where(pd.notnull(df), None)
        
        # 4. Converte datetime in stringhe ISO
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                df[col] = df[col].astype(str)
        
        # 5. Converte tutto in dict records
        return df.to_dict(orient="records")


    # Applica a tutti i tuoi DataFrame
    hourly_sim_clean        = df_to_json_safe(hourly_sim)
    annual_results_df_clean = df_to_json_safe(annual_results_df)
    df_system_clean         = df_to_json_safe(df_system)

    raw_response = {
        "source": "archetype" if archetype else "custom",
        "name": name,
        "category": category,
        "country": country,
        "weather_source": weather_source,
        "validation": {
            "bui_issues": result["bui_issues"],
            "system_messages": result["system_messages"],
        },
        "results": {
            "hourly_building": hourly_sim_clean,
            "annual_building": annual_results_df_clean,
            "hourly_system": df_system_clean,
        },
    }

    return raw_response




@app.post("/report", response_class=HTMLResponse, tags=["Reports"])
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


# @app.post("/simulate_epw", tags=["Simulation"])
# async def simulate_with_epw(
#     epw_file: UploadFile = File(...),
#     bui_json: str = Form(...),
#     system_json: str = Form(...),
# ):
#     """
#     Simulate a building using an EPW weather file.

#     Parameters (multipart/form-data):
#       - epw_file: uploaded EPW file
#       - bui_json: string containing BUI in JSON format
#       - system_json: string containing HVAC system in JSON format

#     Example (curl):

#       curl -X POST http://localhost:8000/simulate_epw \\
#            -F "epw_file=@weather.epw" \\
#            -F 'bui_json={...}' \\
#            -F 'system_json={...}'
#     """
#     try:
#         bui = json.loads(bui_json)
#         system = json.loads(system_json)
#     except json.JSONDecodeError:
#         raise HTTPException(
#             status_code=400,
#             detail="bui_json and system_json must be valid JSON strings.",
#         )

#     bui = numpyfy(bui)
#     system = numpyfy(system)

#     validated = validate_bui_and_system(bui, system)
#     bui_checked = validated["bui_checked"]
#     system_checked = validated["system_checked"]

#     # Save EPW in a temporary directory
#     with tempfile.TemporaryDirectory() as tmpdir:
#         epw_path = os.path.join(tmpdir, epw_file.filename)
#         with open(epw_path, "wb") as f:
#             f.write(await epw_file.read())

#         # Adapt these keyword arguments to match your pybuildingenergy API
#         hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
#             bui_checked,
#             weather_source="epw",
#             weather_file=epw_path,   # Adapt if your function expects a different argument name
#         )

#         calc = pybui.HeatingSystemCalculator(system_checked)
#         calc.load_csv_data(hourly_sim)
#         df_heating_system = calc.run_timeseries()

#     return {
#         "status": "ok",
#         "weather_source": "epw",
#         "hourly_needs": dict_to_df_records(hourly_sim),
#         "annual_needs": dict_to_df_records(annual_results_df),
#         "heating_system": dict_to_df_records(df_heating_system),
#     }


@app.post("/bui/epc_update_u_values", tags=["Greek EPC"])
def update_u_values(
    energy_class: str = Query(
        ...,
        description="Change U-value according to the selected energy class (A, B, C or D) for greek buildings",
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


@app.post("/simulate/batch", tags=["Simulation"])
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

# ================================================================
#                       ECM
# ================================================================

def classify_surface(surface: Dict[str, Any]) -> Optional[str]:
    """
    Ritorna:
      - "roof"     per tetto (opaque, tilt=0, azimuth=0)
      - "wall"     per muri (opaque, tilt=90)
      - "window"   per finestre (transparent, tilt=90)
      - None       altrimenti
    """
    s_type = surface.get("type")
    orientation = surface.get("orientation", {}) or {}
    tilt = orientation.get("tilt")
    azimuth = orientation.get("azimuth")

    # Tetto: opaco, orizzontale (tilt=0) e azimuth=0 (come da tua regola)
    if s_type == "opaque" and tilt == 0 and azimuth == 0:
        return "roof"

    # Muri: opaco, verticale (tilt=90)
    if s_type == "opaque" and tilt == 90:
        return "wall"

    # Finestre: trasparenti, verticali (tilt=90)
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
) -> Dict[str, Any]:
    """
    Ritorna una COPIA del BUI con gli U-value modificati dove richiesto.
    """
    bui_copy = copy.deepcopy(bui)
    for surface in bui_copy.get("building_surface", []):
        kind = classify_surface(surface)
        if kind == "roof" and use_roof and u_roof is not None:
            surface["u_value"] = u_roof
        elif kind == "wall" and use_wall and u_wall is not None:
            surface["u_value"] = u_wall
        elif kind == "window" and use_window and u_window is not None:
            surface["u_value"] = u_window
    return bui_copy

@app.post("/ecm_application", tags=["Simulation"])
async def simulate_uvalues(
    archetype: bool = Query(
        True,
        description="Se True, usa un archetipo (category+country+name). Se False, usa un BUI custom da JSON.",
    ),
    category: Optional[str] = Query(
        None,
        description="Building category. Richiesto quando archetype=True.",
    ),
    country: Optional[str] = Query(
        None,
        description="Country. Richiesto quando archetype=True.",
    ),
    name: Optional[str] = Query(
        None,
        description="Archetype name. Richiesto quando archetype=True.",
    ),
    weather_source: str = Query(
        "pvgis",
        description="Weather source: 'pvgis' o 'epw'.",
    ),
    epw_file: Optional[UploadFile] = File(
        None,
        description="EPW weather file (richiesto quando weather_source='epw').",
    ),
    # BUI custom (quando archetype=False)
    bui_json: Optional[str] = Form(
        None,
        description="JSON string con il BUI (richiesto quando archetype=False).",
    ),
    # Nuovi U-value richiesti
    u_wall: Optional[float] = Query(
        None,
        description="Nuova trasmittanza dei muri (U-value delle superfici opache verticali).",
    ),
    u_roof: Optional[float] = Query(
        None,
        description="Nuova trasmittanza del tetto (U-value della superficie opaca orizzontale).",
    ),
    u_window: Optional[float] = Query(
        None,
        description="Nuova trasmittanza delle finestre (U-value delle superfici trasparenti verticali).",
    ),
):
    """
    Seleziona un BUI da archetipo o da JSON,
    crea copie con U-value modificati per tetto / muri / finestre
    e simula tutte le combinazioni delle migliorie richieste.
    """

    # ---- 1) Selezione BUI base ----
    if archetype:
        if not category or not country or not name:
            raise HTTPException(
                status_code=400,
                detail="Con archetype=true devi fornire 'category', 'country' e 'name'.",
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
                    f"Nessun archetipo trovato per "
                    f"category='{category}', country='{country}', name='{name}'."
                ),
            )

        base_bui = match["bui"]  # BUI di partenza

    else:
        if bui_json is None:
            raise HTTPException(
                status_code=400,
                detail="Con archetype=false devi mandare 'bui_json' come campo form.",
            )
        try:
            bui_raw = json.loads(bui_json)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="'bui_json' deve essere una stringa JSON valida.",
            )

        # Se hai un converter tipo json_to_internal_bui, usalo:
        # base_bui = json_to_internal_bui(bui_raw)
        # Se invece il BUI che arriva è già nel formato interno, puoi fare:
        base_bui = bui_raw

    # ---- 2) Controllo che almeno un U-value sia stato richiesto ----
    u_map = {
        "roof": u_roof,
        "wall": u_wall,
        "window": u_window,
    }
    active_types = [k for k, v in u_map.items() if v is not None]

    if not active_types:
        raise HTTPException(
            status_code=400,
            detail="Devi specificare almeno uno tra u_wall, u_roof, u_window.",
        )

    # ---- 3) Preparazione meteo (EPW) se serve ----
    epw_path: Optional[str] = None
    if weather_source == "epw":
        if epw_file is None:
            raise HTTPException(
                status_code=400,
                detail="Con weather_source='epw' devi caricare 'epw_file'.",
            )
        # salvo il file solo una volta e riuso il path in tutte le simulazioni
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epw") as tmp:
            tmp.write(await epw_file.read())
            epw_path = tmp.name

    elif weather_source != "pvgis":
        raise HTTPException(
            status_code=400,
            detail="weather_source deve essere 'pvgis' oppure 'epw'.",
        )

    # ---- 4) Generazione delle combinazioni di scenari ----
    # tutte le combinazioni non vuote dei tipi attivi
    scenarios_spec = []
    for r in range(1, len(active_types) + 1):
        for subset in itertools.combinations(active_types, r):
            subset = set(subset)
            use_roof = "roof" in subset
            use_wall = "wall" in subset
            use_window = "window" in subset

            label_parts = []
            if use_roof:
                label_parts.append(f"tetto U={u_roof}")
            if use_wall:
                label_parts.append(f"muri U={u_wall}")
            if use_window:
                label_parts.append(f"finestre U={u_window}")
            label = ", ".join(label_parts)

            scenarios_spec.append(
                {
                    "id": "+".join(sorted(subset)),
                    "use_roof": use_roof,
                    "use_wall": use_wall,
                    "use_window": use_window,
                    "label": label,
                }
            )

    # ---- 5) Eseguo le simulazioni per ogni scenario ----
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

            # Lancia ISO 52016
            if weather_source == "pvgis":
                hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                    bui_variant,
                    weather_source="pvgis",
                )
            else:  # EPW
                hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
                    bui_variant,
                    weather_source="epw",
                    path_weather_file=epw_path,
                )

            scenario_results.append(
                {
                    "scenario_id": spec["id"],
                    "description": spec["label"],
                    "u_values": {
                        "roof": u_roof,
                        "wall": u_wall,
                        "window": u_window,
                    },
                    "results": {
                        "hourly_building": clean_and_jsonable(hourly_sim),
                        "annual_building": clean_and_jsonable(annual_results_df),
                    },
                }
            )
    finally:
        # pulizia del file temporaneo EPW se usato
        if epw_path and os.path.exists(epw_path):
            try:
                os.remove(epw_path)
            except OSError:
                pass

    # ---- 6) Risposta ----
    return {
        "source": "archetype" if archetype else "custom",
        "name": name,
        "category": category,
        "country": country,
        "weather_source": weather_source,
        "u_values_requested": {
            "roof": u_roof,
            "wall": u_wall,
            "window": u_window,
        },
        "n_scenarios": len(scenarios_spec),
        "scenarios": scenario_results,
    }


# ==============================================================================
#                      CO2 EQUIVALENT – MODELS, LOGIC & ROUTES
# ==============================================================================

from enum import Enum
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# # Router dedicato alla CO2, allineato con il resto del servizio
# co2_router = APIRouter(
#     prefix="/api",
#     tags=["CO2 emissions"],
# )


# ==============================================================================
#                      FATTORI DI EMISSIONE DATABASE
# ==============================================================================

class EnergySource(str, Enum):
    """Fonti energetiche disponibili"""
    GRID_ELECTRICITY = "grid_electricity"
    NATURAL_GAS = "natural_gas"
    LPG = "lpg"
    DIESEL = "diesel"
    BIOMASS = "biomass"
    DISTRICT_HEATING = "district_heating"
    SOLAR_PV = "solar_pv"
    WIND = "wind"
    HEAT_PUMP_ELECTRIC = "heat_pump_electric"


# Fattori di emissione in kgCO2eq/kWh
EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "IT": {  # Italia
        "grid_electricity": 0.280,      # Mix elettrico nazionale
        "natural_gas": 0.202,           # Gas naturale (uso termico)
        "lpg": 0.234,                   # GPL
        "diesel": 0.267,                # Gasolio da riscaldamento
        "biomass": 0.030,               # Pellet/legna (quasi neutrale)
        "district_heating": 0.180,      # Teleriscaldamento (media)
        "solar_pv": 0.040,              # Fotovoltaico (LCA)
        "wind": 0.012,                  # Eolico (LCA)
        "heat_pump_electric": 0.070,    # PdC con mix elettrico (COP≈4)
    },
    "EU": {  # Media europea
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
    "DE": {  # Germania
        "grid_electricity": 0.420,      # Mix più carbonico
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


# ==============================================================================
#                      PYDANTIC MODELS
# ==============================================================================

class ScenarioInput(BaseModel):
    """Input per calcolo singolo scenario"""
    name: str = Field(..., description="Nome dello scenario")
    energy_source: EnergySource = Field(..., description="Fonte energetica")
    annual_consumption_kwh: float = Field(..., gt=0, description="Consumo annuale [kWh]")
    country: str = Field("IT", description="Codice paese (IT, EU, DE)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Caldaia Gas Esistente",
                "energy_source": "natural_gas",
                "annual_consumption_kwh": 20000,
                "country": "IT",
            }
        }


class MultiScenarioInput(BaseModel):
    """Input per confronto multiplo"""
    scenarios: List[ScenarioInput] = Field(..., min_length=1, max_length=10)

    class Config:
        json_schema_extra = {
            "example": {
                "scenarios": [
                    {
                        "name": "Scenario Attuale - Caldaia Gas",
                        "energy_source": "natural_gas",
                        "annual_consumption_kwh": 20000,
                        "country": "IT",
                    },
                    {
                        "name": "Scenario 1 - Pompa di Calore",
                        "energy_source": "heat_pump_electric",
                        "annual_consumption_kwh": 5000,
                        "country": "IT",
                    },
                    {
                        "name": "Scenario 2 - PdC + Fotovoltaico",
                        "energy_source": "solar_pv",
                        "annual_consumption_kwh": 5000,
                        "country": "IT",
                    },
                ]
            }
        }


class EmissionResult(BaseModel):
    """Risultato calcolo emissioni"""
    name: str
    energy_source: str
    annual_consumption_kwh: float
    emission_factor_kg_per_kwh: float
    annual_emissions_kg_co2eq: float
    annual_emissions_ton_co2eq: float
    equivalent_trees: int          # Alberi necessari per assorbire la CO2
    equivalent_km_car: int         # Km in auto equivalenti

class SavingResult(BaseModel):
    """Risultato di risparmio rispetto al baseline per uno scenario"""
    scenario_name: str
    absolute_kg_co2eq: float
    absolute_ton_co2eq: float
    percentage: float


class ComparisonResult(BaseModel):
    """Risultato confronto scenari"""
    baseline: EmissionResult
    scenarios: List[EmissionResult]
    best_scenario: str
    savings: List[SavingResult]



class InterventionInput(BaseModel):
    """Input per valutare un intervento di riqualificazione"""

    current_consumption_kwh: float = Field(
        ...,
        gt=0,
        description="Consumo attuale [kWh/anno]",
    )
    current_source: EnergySource = Field(
        ...,
        description="Fonte energetica attuale",
    )
    energy_reduction_percentage: float = Field(
        0,
        ge=0,
        le=100,
        description="Riduzione percentuale dei consumi (es. 30 = -30%)",
    )
    new_source: Optional[EnergySource] = Field(
        None,
        description="Nuova fonte energetica (se None, resta quella attuale)",
    )
    new_consumption_kwh: Optional[float] = Field(
        None,
        description="Nuovo consumo annuo. Se None, calcolato dalla riduzione percentuale.",
    )
    country: str = Field(
        "IT",
        description="Codice paese per i fattori di emissione (IT, EU, DE)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "current_consumption_kwh": 20000,
                "current_source": "natural_gas",
                "energy_reduction_percentage": 30,
                "new_source": "heat_pump_electric",
                "new_consumption_kwh": 5000,
                "country": "IT",
            }
        }


# ==============================================================================
#                      BUSINESS LOGIC
# ==============================================================================

def calculate_emissions(
    energy_source: EnergySource | str,
    annual_consumption_kwh: float,
    country: str = "IT",
) -> Dict[str, float]:
    """
    Calcola emissioni CO2eq e parametri equivalenti.
    """
    # Normalizza codice paese
    if country not in EMISSION_FACTORS:
        country = "IT"

    # Normalizza sorgente (Enum -> stringa)
    if isinstance(energy_source, EnergySource):
        source_key = energy_source.value
    else:
        source_key = str(energy_source)

    emission_factor = EMISSION_FACTORS[country].get(
        source_key,
        EMISSION_FACTORS[country]["grid_electricity"],  # fallback ragionevole
    )

    # Calcolo emissioni
    annual_emissions_kg = annual_consumption_kwh * emission_factor
    annual_emissions_ton = annual_emissions_kg / 1000.0

    # EQUIVALENZE PRATICHE
    # 1 albero ≈ 21 kg CO2/anno
    equivalent_trees = int(annual_emissions_kg / 21.0)

    # 1 km in auto ≈ 120 g CO2
    equivalent_km_car = int(annual_emissions_kg / 0.120)

    return {
        "emission_factor": emission_factor,
        "annual_emissions_kg": annual_emissions_kg,
        "annual_emissions_ton": annual_emissions_ton,
        "equivalent_trees": equivalent_trees,
        "equivalent_km_car": equivalent_km_car,
    }


def calculate_savings(
    baseline_emissions: float,
    scenario_emissions: float,
) -> Dict[str, float]:
    """Calcola risparmio tra baseline e scenario."""
    absolute_saving = baseline_emissions - scenario_emissions
    percentage_saving = (
        absolute_saving / baseline_emissions * 100.0
        if baseline_emissions > 0
        else 0.0
    )

    return {
        "absolute_kg_co2eq": round(absolute_saving, 2),
        "absolute_ton_co2eq": round(absolute_saving / 1000.0, 3),
        "percentage": round(percentage_saving, 1),
    }


# ==============================================================================
#                      ROUTES (montate su co2_router)
# ==============================================================================
@app.get("/emission-factors", tags=['Co2 Emissions'])
def get_emission_factors(country: str = "IT"):
    """
    Ottieni fattori di emissione per un paese.

    Args:
        country: Codice paese (IT, EU, DE)
    """
    if country not in EMISSION_FACTORS:
        raise HTTPException(status_code=404, detail=f"Paese {country} non trovato")

    return {
        "country": country,
        "emission_factors_kg_co2eq_per_kwh": EMISSION_FACTORS[country],
        "sources": list(EMISSION_FACTORS[country].keys()),
    }


@app.post("/calculate", response_model=EmissionResult, tags=['Co2 Emissions'])
def calculate_single_scenario(scenario: ScenarioInput):
    """
    Calcola emissioni CO2eq per un singolo scenario.
    """
    result = calculate_emissions(
        energy_source=scenario.energy_source,
        annual_consumption_kwh=scenario.annual_consumption_kwh,
        country=scenario.country,
    )

    return EmissionResult(
        name=scenario.name,
        energy_source=scenario.energy_source,
        annual_consumption_kwh=scenario.annual_consumption_kwh,
        emission_factor_kg_per_kwh=result["emission_factor"],
        annual_emissions_kg_co2eq=result["annual_emissions_kg"],
        annual_emissions_ton_co2eq=result["annual_emissions_ton"],
        equivalent_trees=result["equivalent_trees"],
        equivalent_km_car=result["equivalent_km_car"],
    )


@app.post("/compare", response_model=ComparisonResult, tags=['Co2 Emissions'])
def compare_scenarios(input_data: MultiScenarioInput):
    if len(input_data.scenarios) < 2:
        raise HTTPException(
            status_code=400, 
            detail="Servono almeno 2 scenari per il confronto"
        )
    
    # Calcola emissioni per tutti gli scenari
    results: List[EmissionResult] = []
    for scenario in input_data.scenarios:
        result = calculate_emissions(
            energy_source=scenario.energy_source,
            annual_consumption_kwh=scenario.annual_consumption_kwh,
            country=scenario.country
        )
        
        results.append(EmissionResult(
            name=scenario.name,
            energy_source=scenario.energy_source,
            annual_consumption_kwh=scenario.annual_consumption_kwh,
            emission_factor_kg_per_kwh=result["emission_factor"],
            annual_emissions_kg_co2eq=result["annual_emissions_kg"],
            annual_emissions_ton_co2eq=result["annual_emissions_ton"],
            equivalent_trees=result["equivalent_trees"],
            equivalent_km_car=result["equivalent_km_car"],
        ))
    
    # Baseline = primo scenario
    baseline = results[0]
    baseline_emissions = baseline.annual_emissions_kg_co2eq
    
    # Calcola risparmi
    savings: List[SavingResult] = []
    for result in results[1:]:
        saving_dict = calculate_savings(
            baseline_emissions,
            result.annual_emissions_kg_co2eq
        )
        savings.append(
            SavingResult(
                scenario_name=result.name,
                absolute_kg_co2eq=saving_dict["absolute_kg_co2eq"],
                absolute_ton_co2eq=saving_dict["absolute_ton_co2eq"],
                percentage=saving_dict["percentage"],
            )
        )
    
    # Trova scenario migliore
    best_scenario = min(
        results[1:], 
        key=lambda x: x.annual_emissions_kg_co2eq
    ).name
    
    return ComparisonResult(
        baseline=baseline,
        scenarios=results[1:],   # solo gli alternativi
        best_scenario=best_scenario,
        savings=savings,
    )


@app.post("/calculate-intervention", tags=['Co2 Emissions'])
def calculate_intervention_impact(payload: InterventionInput):
    """
    Calcola impatto di un intervento di riqualificazione energetica
    in termini di CO2 equivalente.
    """
    current_consumption_kwh = payload.current_consumption_kwh
    current_source = payload.current_source
    energy_reduction_percentage = payload.energy_reduction_percentage
    new_source = payload.new_source
    new_consumption_kwh = payload.new_consumption_kwh
    country = payload.country

    # Scenario attuale
    current = calculate_emissions(
        energy_source=current_source,
        annual_consumption_kwh=current_consumption_kwh,
        country=country,
    )

    # Nuovo consumo
    if new_consumption_kwh is None:
        final_consumption = current_consumption_kwh * (
            1.0 - energy_reduction_percentage / 100.0
        )
    else:
        final_consumption = new_consumption_kwh

    # Scenario futuro
    final_source = new_source if new_source is not None else current_source
    future = calculate_emissions(
        energy_source=final_source,
        annual_consumption_kwh=final_consumption,
        country=country,
    )

    # Risparmio
    saving = calculate_savings(
        baseline_emissions=current["annual_emissions_kg"],
        scenario_emissions=future["annual_emissions_kg"],
    )

    return {
        "intervention_summary": {
            "energy_reduction": f"{energy_reduction_percentage}%",
            "source_change": f"{current_source} → {final_source}",
            "consumption_change": (
                f"{current_consumption_kwh:.0f} → {final_consumption:.0f} kWh/anno"
            ),
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
            "trees_saved": (
                current["equivalent_trees"] - future["equivalent_trees"]
            ),
            "km_car_avoided": (
                current["equivalent_km_car"] - future["equivalent_km_car"]
            ),
        },
    }

