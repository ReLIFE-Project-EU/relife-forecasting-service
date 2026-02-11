from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import math
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()


# =============================================================================
# Battery model (optional)
# =============================================================================

@dataclass
class BatteryParams:
    capacity_kwh: float = 10.0
    max_charge_power_kw: float = 5.0
    max_discharge_power_kw: float = 5.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    self_discharge_rate_per_hour: float = 0.0001
    min_soc: float = 0.10
    max_soc: float = 0.90
    initial_soc: float = 0.50


class BatterySimulator:
    def __init__(self, params: BatteryParams):
        self.p = params
        self.soc = params.initial_soc * params.capacity_kwh

    def reset(self):
        self.soc = self.p.initial_soc * self.p.capacity_kwh

    def step(self, net_load_kw: float, timestep_hours: float = 1.0) -> Dict[str, float]:
        # Self-discharge
        self.soc *= (1.0 - self.p.self_discharge_rate_per_hour * timestep_hours)

        energy_kwh = net_load_kw * timestep_hours

        charge_kwh = 0.0
        discharge_kwh = 0.0
        grid_import_kwh = 0.0
        grid_export_kwh = 0.0

        if energy_kwh > 0:
            # deficit -> discharge then grid
            max_discharge_from_soc = min(
                (self.soc - self.p.min_soc * self.p.capacity_kwh),
                self.p.max_discharge_power_kw * timestep_hours,
            )
            max_discharge_from_soc = max(0.0, max_discharge_from_soc)

            # energy delivered to load after discharge efficiency
            discharge_kwh = min(energy_kwh, max_discharge_from_soc * self.p.discharge_efficiency)
            discharge_from_soc = discharge_kwh / self.p.discharge_efficiency if self.p.discharge_efficiency > 0 else 0.0

            self.soc -= discharge_from_soc
            grid_import_kwh = max(0.0, energy_kwh - discharge_kwh)

        else:
            # surplus -> charge then export
            surplus_kwh = -energy_kwh
            max_charge_space = self.p.max_soc * self.p.capacity_kwh - self.soc
            max_charge_from_grid = min(
                max_charge_space,
                self.p.max_charge_power_kw * timestep_hours,
            )
            max_charge_from_grid = max(0.0, max_charge_from_grid)

            charge_to_battery = min(surplus_kwh, max_charge_from_grid)
            charge_kwh = charge_to_battery * self.p.charge_efficiency

            self.soc += charge_kwh
            grid_export_kwh = max(0.0, surplus_kwh - charge_to_battery)

        # clamp SOC
        self.soc = max(
            self.p.min_soc * self.p.capacity_kwh,
            min(self.p.max_soc * self.p.capacity_kwh, self.soc),
        )

        return {
            "battery_charge_kwh": float(charge_kwh) if charge_kwh > 1e-9 else 0.0,
            "battery_discharge_kwh": float(discharge_kwh) if discharge_kwh > 1e-9 else 0.0,
            "grid_import_kwh": float(grid_import_kwh) if grid_import_kwh > 1e-9 else 0.0,
            "grid_export_kwh": float(grid_export_kwh) if grid_export_kwh > 1e-9 else 0.0,
            "soc_kwh": float(self.soc),
            "soc_fraction": float(self.soc / self.p.capacity_kwh) if self.p.capacity_kwh > 0 else 0.0,
        }


# =============================================================================
# PV profile
# =============================================================================

def _simple_pv_hourly_profile(
    latitude: float,
    hours_per_year: int,
    annual_yield_kwh_per_kwp: float = 1400.0,
    tilt_deg: float = 30.0,
    azimuth_deg: float = 0.0,  # 0=south, 90=west, -90=east
) -> np.ndarray:
    """
    VERY simplified PV profile [kWh/kWp per hour]. Use PVGIS in production if possible.
    Normalized to annual_yield_kwh_per_kwp.
    """
    prof = np.zeros(hours_per_year)

    lat_rad = math.radians(latitude)
    tilt_rad = math.radians(tilt_deg)
    az_rad = math.radians(azimuth_deg)

    for h in range(hours_per_year):
        day = h // 24
        hour = h % 24

        decl = 23.45 * math.sin(math.radians(360 * (284 + day) / 365))
        dec_rad = math.radians(decl)

        hour_angle = math.radians(15 * (hour - 12))
        sin_alt = (math.sin(lat_rad) * math.sin(dec_rad) +
                   math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_angle))

        if sin_alt <= 0:
            continue

        altitude = math.asin(max(-1.0, min(1.0, sin_alt)))

        cos_inc = (math.sin(altitude) * math.cos(tilt_rad) +
                   math.cos(altitude) * math.sin(tilt_rad) * math.cos(az_rad))
        cos_inc = max(0.0, cos_inc)

        air_mass = 1.0 / max(0.01, sin_alt)
        ghi_clear = 1000 * sin_alt * (0.7 ** (air_mass ** 0.678))

        poa = ghi_clear * cos_inc / max(0.01, sin_alt)
        cloud = 0.6 + 0.2 * math.cos(math.radians(360 * day / 365))

        eta = 0.15
        prof[h] = max(0.0, poa * eta * cloud / 1000.0)

    s = prof.sum()
    if s > 0:
        prof *= (annual_yield_kwh_per_kwp / s)
    return prof


def _fetch_pvgis_hourly_pv_kwh_per_kwp(
    latitude: float,
    longitude: float,
    tilt_deg: float = 30.0,
    azimuth_deg: float = 0.0,
    loss_percent: float = 14.0,
    pv_tech: str = "crystSi",
    year: Optional[int] = None,
) -> np.ndarray:
    """
    PVGIS seriescalc hourly PV for 1 kWp -> returns kWh/kWp per hour.

    Notes:
    - PVGIS returns hourly list with 'P' (W) for PV power (depending on output fields).
    - We integrate as 1-hour timestep: kWh = P(W)/1000.
    """
    base = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    params = {
        "lat": latitude,
        "lon": longitude,
        "outputformat": "json",
        "pvcalculation": 1,
        "peakpower": 1,               # 1 kWp -> output is per-kWp (then scale)
        "loss": loss_percent,         # system losses [%]
        "angle": tilt_deg,
        "aspect": azimuth_deg,        # PVGIS: aspect degrees from south? commonly: 0 south, 90 west, -90 east
        "pvtechchoice": pv_tech,
        "mountingplace": "building",
        "optimalangles": 0,
        "optimalinclination": 0,
        "components": 0,
        "usehorizon": 1,
    }
    if year is not None:
        params["startyear"] = int(year)
        params["endyear"] = int(year)

    url = base + "?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"PVGIS request failed: {exc}") from exc

    try:
        hourly = data["outputs"]["hourly"]
    except Exception as exc:
        raise RuntimeError("PVGIS response missing outputs.hourly") from exc

    # Prefer 'P' if available
    p_w = []
    for row in hourly:
        if "P" in row:
            p_w.append(float(row["P"]))
        elif "power" in row:
            p_w.append(float(row["power"]))
        else:
            # If no power key, fail loudly (caller may fallback)
            raise RuntimeError("PVGIS hourly rows do not include PV power key ('P').")

    p_w = np.asarray(p_w, dtype=float)
    # 1h timestep => kWh = W/1000
    kwh = p_w / 1000.0
    return kwh


# =============================================================================
# Input extraction: UNI/TS 11300 endpoint output
# =============================================================================

def _extract_uni_hourly_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Expected to receive the output of /primary-energy/uni11300, or an object containing it.

    Supported shapes:
      - {"hourly_results": [...]}
      - {"uni11300": {"hourly_results": [...]} }
      - {"results": {"primary_energy_uni11300": {"hourly_results": [...]}}}
      - {"primary_energy_uni11300": {"hourly_results": [...]}}
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object.")

    cand = None
    if "hourly_results" in payload:
        cand = payload["hourly_results"]
    elif isinstance(payload.get("uni11300"), dict) and "hourly_results" in payload["uni11300"]:
        cand = payload["uni11300"]["hourly_results"]
    elif isinstance(payload.get("primary_energy_uni11300"), dict) and "hourly_results" in payload["primary_energy_uni11300"]:
        cand = payload["primary_energy_uni11300"]["hourly_results"]
    elif isinstance(payload.get("results"), dict):
        pe = payload["results"].get("primary_energy_uni11300")
        if isinstance(pe, dict) and "hourly_results" in pe:
            cand = pe["hourly_results"]

    if cand is None:
        raise HTTPException(
            status_code=400,
            detail="Missing UNI/TS 11300 hourly data. Provide /primary-energy/uni11300 response with 'hourly_results'.",
        )
    if not isinstance(cand, list) or not cand:
        raise HTTPException(status_code=400, detail="'hourly_results' must be a non-empty list.")

    df = pd.DataFrame(cand)
    if df.empty:
        raise HTTPException(status_code=400, detail="UNI hourly_results parsed empty.")

    # If timestamp exists, use it as index (recommended)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise HTTPException(status_code=400, detail="Invalid timestamps in UNI hourly_results.")
        df = df.set_index("timestamp").sort_index()

    return df


def _get_hp_electric_series_kwh(df_uni: pd.DataFrame) -> pd.Series:
    """
    Prefer UNI fields (already post subsystem losses):
      - E_delivered_electric_total_kWh
    fallback:
      - E_delivered_electric_heat_kWh + E_delivered_electric_cool_kWh
    """
    if "E_delivered_electric_total_kWh" in df_uni.columns:
        s = pd.to_numeric(df_uni["E_delivered_electric_total_kWh"], errors="coerce").fillna(0.0)
        return s.clip(lower=0.0)

    heat = pd.to_numeric(df_uni.get("E_delivered_electric_heat_kWh", 0.0), errors="coerce").fillna(0.0)
    cool = pd.to_numeric(df_uni.get("E_delivered_electric_cool_kWh", 0.0), errors="coerce").fillna(0.0)
    s = heat + cool
    return s.clip(lower=0.0)


# =============================================================================
# Core analysis
# =============================================================================

def analyze_pv_hp_from_uni11300(
    uni_payload: Dict[str, Any],
    pv_kwp: float,
    latitude: float,
    longitude: float,
    tilt_deg: float = 30.0,
    azimuth_deg: float = 0.0,
    use_pvgis: bool = True,
    annual_pv_yield_kwh_per_kwp: float = 1400.0,
    battery_params: Optional[BatteryParams] = None,
    pvgis_loss_percent: float = 14.0,
    pvgis_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Uses UNI/TS 11300 hourly results as HP electricity demand input.
    Computes PV generation (PVGIS or simplified), optional battery, grid import/export,
    self-consumption and self-sufficiency.

    Output:
      - summary (annual)
      - hourly (optional, returned as records)
    """
    if pv_kwp <= 0:
        raise HTTPException(status_code=400, detail="pv_kwp must be > 0.")
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        raise HTTPException(status_code=400, detail="Invalid latitude/longitude.")

    df_uni = _extract_uni_hourly_df(uni_payload)
    hp_elec_kwh = _get_hp_electric_series_kwh(df_uni)

    n_hours = len(hp_elec_kwh)
    if n_hours < 24:
        raise HTTPException(status_code=400, detail=f"Too few hours in UNI hourly_results: {n_hours}")

    # PV profile per kWp (kWh/kWp per hour)
    pv_per_kwp = None
    pv_source = None
    if use_pvgis:
        try:
            pv_per_kwp = _fetch_pvgis_hourly_pv_kwh_per_kwp(
                latitude=latitude,
                longitude=longitude,
                tilt_deg=tilt_deg,
                azimuth_deg=azimuth_deg,
                loss_percent=pvgis_loss_percent,
                year=pvgis_year,
            )
            pv_source = "pvgis"
        except Exception:
            pv_per_kwp = None

    if pv_per_kwp is None:
        pv_per_kwp = _simple_pv_hourly_profile(
            latitude=latitude,
            hours_per_year=n_hours,
            annual_yield_kwh_per_kwp=annual_pv_yield_kwh_per_kwp,
            tilt_deg=tilt_deg,
            azimuth_deg=azimuth_deg,
        )
        pv_source = "simple"

    if len(pv_per_kwp) != n_hours:
        # PVGIS typically returns 8760. If UNI has leap year (8784) or different length, fallback to simple.
        pv_per_kwp = _simple_pv_hourly_profile(
            latitude=latitude,
            hours_per_year=n_hours,
            annual_yield_kwh_per_kwp=annual_pv_yield_kwh_per_kwp,
            tilt_deg=tilt_deg,
            azimuth_deg=azimuth_deg,
        )
        pv_source = "simple_resampled"

    pv_gen_kwh = pv_per_kwp * float(pv_kwp)

    # Battery
    battery = BatterySimulator(battery_params) if battery_params else None

    grid_import = np.zeros(n_hours)
    grid_export = np.zeros(n_hours)
    self_cons = np.zeros(n_hours)
    batt_ch = np.zeros(n_hours)
    batt_dis = np.zeros(n_hours)
    batt_soc = np.zeros(n_hours)
    net_load = np.zeros(n_hours)

    hp_vals = hp_elec_kwh.to_numpy(dtype=float)

    for h in range(n_hours):
        load = hp_vals[h]
        pv = float(pv_gen_kwh[h])

        # net >0 means deficit
        net = load - pv
        net_load[h] = net

        if battery:
            # battery.step expects kW; timestep 1h => kW == kWh/h
            step = battery.step(net_load_kw=net, timestep_hours=1.0)
            grid_import[h] = step["grid_import_kwh"]
            grid_export[h] = step["grid_export_kwh"]
            batt_ch[h] = step["battery_charge_kwh"]
            batt_dis[h] = step["battery_discharge_kwh"]
            batt_soc[h] = step["soc_kwh"]

            # PV used directly + battery discharge to cover load (battery only charged from surplus PV in this model)
            self_cons[h] = min(load, pv) + batt_dis[h]
        else:
            if net > 0:
                grid_import[h] = net
                self_cons[h] = pv
            else:
                grid_export[h] = -net
                self_cons[h] = load

    df_out = pd.DataFrame(
        {
            "hp_electric_kWh": hp_vals,
            "pv_generation_kWh": pv_gen_kwh,
            "net_load_kWh": net_load,
            "self_consumption_kWh": self_cons,
            "grid_import_kWh": grid_import,
            "grid_export_kWh": grid_export,
            "battery_charge_kWh": batt_ch,
            "battery_discharge_kWh": batt_dis,
            "battery_soc_kWh": batt_soc,
        },
        index=df_uni.index,
    )

    annual_hp = float(df_out["hp_electric_kWh"].sum())
    annual_pv = float(df_out["pv_generation_kWh"].sum())
    annual_self = float(df_out["self_consumption_kWh"].sum())
    annual_imp = float(df_out["grid_import_kWh"].sum())
    annual_exp = float(df_out["grid_export_kWh"].sum())

    self_consumption_rate = (annual_self / annual_pv) if annual_pv > 0 else 0.0
    self_sufficiency_rate = (annual_self / annual_hp) if annual_hp > 0 else 0.0

    summary = {
        "inputs": {
            "pv_kwp": float(pv_kwp),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "tilt_deg": float(tilt_deg),
            "azimuth_deg": float(azimuth_deg),
            "pv_profile_source": pv_source,
            "n_hours": int(n_hours),
            "has_battery": battery is not None,
            "battery_params": (battery_params.__dict__ if battery_params else None),
        },
        "annual_kwh": {
            "hp_electric": annual_hp,
            "pv_generation": annual_pv,
            "self_consumption": annual_self,
            "grid_import": annual_imp,
            "grid_export": annual_exp,
        },
        "indicators": {
            "self_consumption_rate": float(self_consumption_rate),
            "self_sufficiency_rate": float(self_sufficiency_rate),
        },
    }

    return {
        "summary": summary,
        "hourly_results": df_out.reset_index().to_dict(orient="records"),
    }


# =============================================================================
# FastAPI endpoint
# =============================================================================

@router.post("/pv-hp/analysis", tags=["PV + Heat Pump"])
def pv_hp_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute PV+HP hourly matching using UNI/TS 11300 endpoint output as input.

    Required:
      - 'uni11300': (or a shape containing it) must include 'hourly_results'
      - pv_kwp, latitude, longitude

    Optional:
      - tilt_deg, azimuth_deg
      - use_pvgis (default True)
      - pvgis_loss_percent (default 14)
      - pvgis_year (optional)
      - annual_pv_yield_kwh_per_kwp (fallback profile only)
      - battery_params: {capacity_kwh, ...}
    """
    pv_kwp = payload.get("pv_kwp")
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")

    if pv_kwp is None or latitude is None or longitude is None:
        raise HTTPException(status_code=400, detail="Missing required fields: pv_kwp, latitude, longitude.")

    tilt_deg = float(payload.get("tilt_deg", 30.0))
    azimuth_deg = float(payload.get("azimuth_deg", 0.0))
    use_pvgis = bool(payload.get("use_pvgis", True))
    annual_yield = float(payload.get("annual_pv_yield_kwh_per_kwp", 1400.0))
    pvgis_loss = float(payload.get("pvgis_loss_percent", 14.0))
    pvgis_year = payload.get("pvgis_year", None)
    pvgis_year = int(pvgis_year) if pvgis_year is not None else None

    battery_params_obj = None
    if payload.get("battery_params") is not None:
        if not isinstance(payload["battery_params"], dict):
            raise HTTPException(status_code=400, detail="'battery_params' must be a JSON object.")
        try:
            battery_params_obj = BatteryParams(**payload["battery_params"])
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid battery_params: {exc}") from exc

    # UNI payload can be nested
    uni_payload = payload.get("uni11300", payload)

    return analyze_pv_hp_from_uni11300(
        uni_payload=uni_payload,
        pv_kwp=float(pv_kwp),
        latitude=float(latitude),
        longitude=float(longitude),
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        use_pvgis=use_pvgis,
        annual_pv_yield_kwh_per_kwp=annual_yield,
        battery_params=battery_params_obj,
        pvgis_loss_percent=pvgis_loss,
        pvgis_year=pvgis_year,
    )
