from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel, Field


class Project:
    def __init__(self):
        self.building: Optional[Dict[str, Any]] = None
        self.plant: Optional[Dict[str, Any]] = None
        self.results: Optional[pd.DataFrame] = None


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
                "cool_efficiency": 3.0,  # EER/COP per raffrescamento
            }
        ],
    )
