from __future__ import annotations

import io
import uuid
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Building Thermal Simulator & EPC API (demo)")

# -------------------------------
# In-memory storage (demo only)
# -------------------------------
class Project:
    def __init__(self):
        self.building: Optional[Dict[str, Any]] = None
        self.plant: Optional[Dict[str, Any]] = None
        self.results: Optional[pd.DataFrame] = None

PROJECTS: Dict[str, Project] = {}

# -------------------------------
# Pydantic models
# -------------------------------
class BuildingPayload(BaseModel):
    data: Dict[str, Any] = Field(
        ...,
        description=(
            "Dizionario con parametri geometrici/termici. Esempio chiavi: \n"
            "- area_m2 (float)\n- volume_m3 (float)\n- U_envelope_W_m2K (float)\n"
            "- infiltration_ach (float/h)\n- internal_gains_W (float)\n- thermal_capacity_kJ_K (float)\n"
        ),
        examples=[
            {
                "area_m2": 100.0,
                "volume_m3": 250.0,
                "U_envelope_W_m2K": 0.7,
                "infiltration_ach": 0.5,
                "internal_gains_W": 500.0,
                "thermal_capacity_kJ_K": 80000.0,
            }
        ],
    )

class PlantPayload(BaseModel):
    data: Dict[str, Any] = Field(
        ...,
        description=(
            "Dizionario impianto. Esempio chiavi: \n"
            "- heat_setpoint_C (float)\n- cool_setpoint_C (float)\n- heat_power_max_W (float)\n- cool_power_max_W (float)\n- heat_efficiency (float, 0-1 o COP)\n- cool_efficiency (float, 0-1 o EER)\n"
        ),
        examples=[
            {
                "heat_setpoint_C": 20.0,
                "cool_setpoint_C": 26.0,
                "heat_power_max_W": 6000.0,
                "cool_power_max_W": 6000.0,
                "heat_efficiency": 0.95,  # caldaia (rendimento)
                "cool_efficiency": 3.0,   # EER/COP per raffrescamento
            }
        ],
    )

# -------------------------------
# Helpers
# -------------------------------

def new_project_id() -> str:
    pid = str(uuid.uuid4())
    PROJECTS[pid] = Project()
    return pid


def get_project(pid: str) -> Project:
    prj = PROJECTS.get(pid)
    if prj is None:
        raise HTTPException(status_code=404, detail="project_id non trovato")
    return prj


# EPW columns: we only use dry-bulb temperature (°C).
# EPW has 8 header lines; data lines are CSV with many columns. Dry bulb is column index 6 (0-based) in EnergyPlus EPW.
# We'll do a defensive parse and try common positions.

def parse_epw_hours(epw_bytes: bytes) -> pd.Series:
    df = pd.read_csv(io.BytesIO(epw_bytes), header=None, skiprows=8)
    # Try EnergyPlus standard index for dry bulb
    for idx in (6, 7):  # 6 typical, 7 fallback if format shifted
        if idx < df.shape[1]:
            s = df.iloc[:, idx]
            # Very rough sanity: temperatures in [-60, 60]
            if s.dropna().between(-80, 80).mean() > 0.9:
                t_out = s.astype(float).reset_index(drop=True)
                t_out.index.name = "hour"
                return t_out
    # As last resort, take the column with most values in plausible range
    plausible = df.apply(lambda c: c.dropna().between(-80, 80).mean())
    best = plausible.idxmax()
    t_out = df.iloc[:, best].astype(float).reset_index(drop=True)
    t_out.index.name = "hour"
    return t_out


def run_rc_simulation(building: Dict[str, Any], plant: Dict[str, Any], t_outdoor: pd.Series) -> pd.DataFrame:
    """Semplice modello 1R1C con controllo on/off per mantenere i setpoint.

    Equazione discreta oraria:
        T_{k+1} = T_k + (dt/C) * [ UA*(T_out - T_k) + Q_int + Q_HVAC ]

    dove:
      - dt = 3600 s
      - C = capacità termica [J/K]
      - UA = U_envelope * area [W/K]
      - Q_int = carichi interni [W]
      - Q_HVAC è positivo in riscaldamento, negativo in raffrescamento (lim. di potenza)

    Consumi:
      - E_heat = max(Q_HVAC, 0) / heat_efficiency [Wh]
      - E_cool = max(-Q_HVAC, 0) / cool_efficiency [Wh]
    """
    area = float(building.get("area_m2", 100.0))
    U = float(building.get("U_envelope_W_m2K", 0.9))
    UA = U * area  # W/K
    V = float(building.get("volume_m3", area * 2.5))
    ach = float(building.get("infiltration_ach", 0.5))
    rho_air = 1.2  # kg/m3
    cp_air = 1005  # J/kg-K
    # Infiltration conductance approx: m_dot * cp, with m_dot = rho * V * ach / 3600
    G_inf = rho_air * V * ach / 3600.0 * cp_air  # W/K

    UA_total = UA + G_inf

    C_kJ = float(building.get("thermal_capacity_kJ_K", 80000.0))
    C = C_kJ * 1000  # to J/K

    Q_int = float(building.get("internal_gains_W", 500.0))

    # Plant params
    T_heat = float(plant.get("heat_setpoint_C", 20.0))
    T_cool = float(plant.get("cool_setpoint_C", 26.0))
    P_heat = float(plant.get("heat_power_max_W", 6000.0))
    P_cool = float(plant.get("cool_power_max_W", 6000.0))
    eff_h = float(plant.get("heat_efficiency", 0.95))  # rendimento o COP
    eff_c = float(plant.get("cool_efficiency", 3.0))   # EER/COP

    dt = 3600.0
    n = len(t_outdoor)
    T = pd.Series(index=range(n), dtype=float)
    Qhvac = pd.Series(index=range(n), dtype=float)
    E_heat_Wh = pd.Series(index=range(n), dtype=float)
    E_cool_Wh = pd.Series(index=range(n), dtype=float)

    # Stato iniziale: 0 -> adotta il primo setpoint o la T esterna
    T0 = t_outdoor.iloc[0]
    T[0] = min(max(T0, T_heat), T_cool)
    Qhvac[0] = 0.0
    E_heat_Wh[0] = 0.0
    E_cool_Wh[0] = 0.0

    for k in range(n - 1):
        T_k = T[k]
        T_out = float(t_outdoor.iloc[k])

        # Controllo HVAC semplice
        q_hvac = 0.0
        if T_k < T_heat:  # riscaldamento
            q_hvac = min(P_heat, (T_heat - T_k) * UA_total)  # limitazione semplificata
        elif T_k > T_cool:  # raffrescamento
            q_hvac = -min(P_cool, (T_k - T_cool) * UA_total)

        # Aggiornamento stato
        dT = (dt / C) * (UA_total * (T_out - T_k) + Q_int + q_hvac)
        T[k + 1] = T_k + dT
        Qhvac[k] = q_hvac

        # Consumi
        if q_hvac >= 0:
            E_heat_Wh[k] = q_hvac / max(eff_h, 1e-6) * (dt / 3600.0)
            E_cool_Wh[k] = 0.0
        else:
            E_cool_Wh[k] = (-q_hvac) / max(eff_c, 1e-6) * (dt / 3600.0)
            E_heat_Wh[k] = 0.0

    # Ultimo step consumi = 0 by definition for missing q_hvac computation
    Qhvac.iloc[-1] = Qhvac.iloc[-2]
    E_heat_Wh.iloc[-1] = 0.0
    E_cool_Wh.iloc[-1] = 0.0

    df = pd.DataFrame(
        {
            "T_out_C": t_outdoor.values,
            "T_in_C": T.values,
            "Q_HVAC_W": Qhvac.values,
            "E_heat_Wh": E_heat_Wh.values,
            "E_cool_Wh": E_cool_Wh.values,
        }
    )
    df.index.name = "hour"
    return df


def default_plant_template() -> Dict[str, Any]:
    return {
        "heat_setpoint_C": 20.0,
        "cool_setpoint_C": 26.0,
        "heat_power_max_W": 6000.0,
        "cool_power_max_W": 6000.0,
        "heat_efficiency": 0.95,
        "cool_efficiency": 3.0,
    }


def compute_epc(building: Dict[str, Any], results: pd.DataFrame) -> Dict[str, Any]:
    """Trasforma il risultato in un EPC semplificato usando input di default.

    Metodo:
    - Calcolo energia utile (Wh) riscaldamento/raffrescamento -> energia primaria semplificata
    - Normalizzazione per area (kWh/m2 anno)
    - Mappatura a classi EPC fittizie (A4...G) su soglie predefinite.
    """
    area = float(building.get("area_m2", 100.0))
    # Conversione a kWh
    E_heat_kWh = results["E_heat_Wh"].sum() / 1000.0
    E_cool_kWh = results["E_cool_Wh"].sum() / 1000.0

    # Fattori di conversione a energia primaria (semplificati)
    fp_heat = 1.0  # es. gas naturale ~1.0-1.05 (sempl.)
    fp_elec = 2.2  # elettrico semplificato

    # Supponiamo: riscaldamento da caldaia (fp_heat), raffrescamento elettrico (fp_elec)
    EP_heat = E_heat_kWh * fp_heat
    EP_cool = E_cool_kWh * fp_elec
    EP_tot_kWh = EP_heat + EP_cool

    EUI = EP_tot_kWh / max(area, 1e-6)  # kWh/m2*y

    # Soglie fittizie per classi (esempio indicativo)
    thresholds = [
        (20, "A4"), (30, "A3"), (40, "A2"), (50, "A1"), (60, "B"),
        (80, "C"), (110, "D"), (140, "E"), (180, "F"), (10**9, "G"),
    ]
    for thr, label in thresholds:
        if EUI <= thr:
            epc_class = label
            break

    return {
        "area_m2": area,
        "E_heat_kWh": round(EP_heat, 2),
        "E_cool_kWh": round(EP_cool, 2),
        "EP_tot_kWh": round(EP_tot_kWh, 2),
        "EUI_kWh_m2y": round(EUI, 2),
        "class": epc_class,
        "method": "Semplificato con fattori di conversione di default",
    }


# Placeholder for external simulator integration (e.g., pybuildingenergy)
# def run_external_simulator(building: Dict[str, Any], plant: Dict[str, Any], epw_path: str) -> pd.DataFrame:
#     """Integra qui la libreria esterna, restituendo un DataFrame orario."""
#     raise NotImplementedError


# -------------------------------
# API Endpoints
# -------------------------------

@app.post("/project")
def create_project() -> Dict[str, str]:
    pid = new_project_id()
    return {"project_id": pid}


@app.put("/project/{project_id}/building")
def upload_building(project_id: str, payload: BuildingPayload):
    prj = get_project(project_id)
    prj.building = payload.data
    return {"status": "ok", "project_id": project_id, "building_keys": list(payload.data.keys())}


@app.get("/plant/template")
def get_plant_template():
    return default_plant_template()


@app.put("/project/{project_id}/plant")
def upload_plant(project_id: str, payload: PlantPayload):
    prj = get_project(project_id)
    prj.plant = payload.data
    return {"status": "ok", "project_id": project_id, "plant_keys": list(payload.data.keys())}


@app.post("/project/{project_id}/simulate")
def simulate_project(project_id: str, epw: UploadFile = File(...)):
    prj = get_project(project_id)
    if prj.building is None:
        raise HTTPException(status_code=400, detail="Caricare prima l'edificio")
    if prj.plant is None:
        # Se non caricato, usa template di default
        prj.plant = default_plant_template()

    if not epw.filename.lower().endswith(".epw"):
        raise HTTPException(status_code=400, detail="Fornire un file EPW valido")

    epw_bytes = epw.file.read()

    try:
        t_out = parse_epw_hours(epw_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore lettura EPW: {e}")

    if len(t_out) == 0:
        raise HTTPException(status_code=400, detail="EPW vuoto o non valido")

    # Se si volesse usare un simulatore esterno, sostituire con:
    # results = run_external_simulator(prj.building, prj.plant, epw_saved_path)
    results = run_rc_simulation(prj.building, prj.plant, t_out)

    prj.results = results
    return {"status": "ok", "n_hours": int(results.shape[0])}


@app.get("/project/{project_id}/results.csv")
def download_results_csv(project_id: str):
    prj = get_project(project_id)
    if prj.results is None:
        raise HTTPException(status_code=404, detail="Nessun risultato disponibile. Eseguire la simulazione.")

    csv_buf = io.StringIO()
    prj.results.to_csv(csv_buf)
    csv_buf.seek(0)
    return StreamingResponse(
        io.BytesIO(csv_buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=results_{project_id}.csv"},
    )


@app.get("/project/{project_id}/epc")
def get_epc(project_id: str):
    prj = get_project(project_id)
    if prj.results is None or prj.building is None:
        raise HTTPException(status_code=400, detail="Richiede edificio e risultati di simulazione")

    epc = compute_epc(prj.building, prj.results)
    return JSONResponse(epc)


# Healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}
