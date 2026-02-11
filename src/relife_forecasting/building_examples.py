# relife_forecasting/building_examples.py
# -----------------------------------------------------------------------------
# Clean, JSON-serializable archetype inputs compatible with main.py endpoints:
#   - /building (archetype=true)
#   - /validate (archetype=true)
#   - /simulate (archetype=true)
#   - /ecm_application/run_sequential_save (archetype=true)
#   - /run/iso52016-uni11300-pv  (this uses only BUI directly, but keeping archetypes is fine)
#
# Key points:
# - NO pandas.DataFrame objects (FastAPI/JSON can't serialize them)
# - Keep everything as dict/list/str/float/int/bool/None
# - Provide BUILDING_ARCHETYPES + UNI11300_SIMULATION_EXAMPLE as expected by main.py
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# BUI: JSON-safe (your original, lightly cleaned)
# -----------------------------------------------------------------------------
BUI_SINGLE_FAMILY_1946_1969: Dict[str, Any] = {
    "building": {
        "name": "test-cy",
        "azimuth_relative_to_true_north": 41.8,
        "latitude": 37.98880066730187,
        "longitude": 23.733531819066098,
        "exposed_perimeter": 46,
        "height": 2.7,
        "wall_thickness": 0.3,
        "n_floors": 2,
        "building_type_class": "Residential_apartment",
        "adj_zones_present": False,
        "number_adj_zone": 0,
        "net_floor_area": 125,
        "construction_class": "class_i",
    },
    "building_surface": [
        {
            "name": "Roof surface",
            "type": "opaque",
            "area": 120,
            "sky_view_factor": 1.0,
            "u_value": 3.05,
            "solar_absorptance": 0.4,
            "thermal_capacity": 741500.0,
            "orientation": {"azimuth": 0, "tilt": 0},
            "name_adj_zone": None,
        },
        {
            "name": "Opaque north surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 0.95,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {"azimuth": 0, "tilt": 90},
            "name_adj_zone": None,
        },
        {
            "name": "Opaque south surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 0.95,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {"azimuth": 180, "tilt": 90},
            "name_adj_zone": None,
        },
        {
            "name": "Opaque east surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 0.95,
            "solar_absorptance": 0.6,
            "thermal_capacity": 1416240.0,
            "orientation": {"azimuth": 90, "tilt": 90},
            "name_adj_zone": None,
        },
        {
            "name": "Opaque west surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 0.95,
            "solar_absorptance": 0.7,
            "thermal_capacity": 1416240.0,
            "orientation": {"azimuth": 270, "tilt": 90},
            "name_adj_zone": None,
        },
        {
            "name": "Slab to ground",
            "type": "opaque",
            "area": 120,
            "sky_view_factor": 0.0,
            "u_value": 1.2,
            "solar_absorptance": 0.6,
            "thermal_capacity": 405801,
            "orientation": {"azimuth": 0, "tilt": 0},
            "name_adj_zone": None,
        },
        {
            "name": "Transparent east surface",
            "type": "transparent",
            "area": 4,
            "sky_view_factor": 0.5,
            "u_value": 3.1,
            "g_value": 0.76,
            "height": 2,
            "width": 1,
            "parapet": 1.1,
            "orientation": {"azimuth": 90, "tilt": 90},
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
            "name_adj_zone": None,
        },
        {
            "name": "Transparent west surface",
            "type": "transparent",
            "area": 4,
            "sky_view_factor": 0.5,
            "u_value": 5.0,
            "g_value": 0.726,
            "height": 2,
            "width": 1,
            "parapet": 1.1,
            "orientation": {"azimuth": 270, "tilt": 90},
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
            "name_adj_zone": None,
        },
    ],
    "units": {
        "area": "m²",
        "u_value": "W/m²K",
        "thermal_capacity": "J/kgK",
        "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
        "tilt": "degrees (0=horizontal, 90=vertical)",
        "internal_gain": "W/m²",
        "internal_gain_profile": "Normalized to 0-1",
        "HVAC_profile": "0: off, 1: on",
    },
    "building_parameters": {
        "temperature_setpoints": {
            "heating_setpoint": 20.0,
            "heating_setback": 17.0,
            "cooling_setpoint": 26.0,
            "cooling_setback": 30.0,
            "units": "°C",
        },
        "system_capacities": {
            "heating_capacity": 10000000.0,
            "cooling_capacity": 12000000.0,
            "units": "W",
        },
        "airflow_rates": {
            "infiltration_rate": 1.0,
            "units": "ACH (air changes per hour)",
        },
        "internal_gains": [
            {
                "name": "occupants",
                "full_load": 4.2,
                "weekday": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 1.0, 1.0],
                "weekend": [1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0],
            },
            {
                "name": "appliances",
                "full_load": 3.0,
                "weekday": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6],
                "weekend": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6],
            },
            {
                "name": "lighting",
                "full_load": 3.0,
                "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15],
                "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15],
            },
        ],
        "construction": {
            "wall_thickness": 0.3,
            "thermal_bridges": 0.8,
            "units": "m (for thickness), W/mK (for thermal bridges)",
        },
        "climate_parameters": {"coldest_month": 1, "units": "1-12 (January-December)"},
        "heating_profile": {
            "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        },
        "cooling_profile": {
            "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        },
        "ventilation_profile": {
            "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        },
    },
}

# -----------------------------------------------------------------------------
# SYSTEM: JSON-safe (condensing boiler + radiators)
# NOTE: fields that were DataFrames are now list-of-dict.
# -----------------------------------------------------------------------------
SYSTEM_CONDENSING_BOILER_RADIATORS: Dict[str, Any] = {
    # ---- emission ----
    "emitter_type": "Radiator",
    "nominal_power": 10,  # kW
    "emission_efficiency": 95,  # %
    "selected_emm_cont_circuit": 1,
    "flow_temp_control_type": "Type 2 - Based on outdoor temperature",
    "mixing_valve": False,
    "mixing_valve_delta": 0,

    # previously pd.DataFrame -> list-of-dict
    "heat_emission_data": [
        {
            "θH_em_flw_max_sahz_i": 70,
            "ΔθH_em_w_max_sahz_i": 20,
            "θH_em_ret_req_sahz_i": 45,
            "βH_em_req_sahz_i": 80,
            "θH_em_flw_min_tz_i": 30,
        }
    ],
    "outdoor_temp_data": [
        {
            "θext_min_sahz_i": -10,
            "θext_max_sahz_i": 16,
            "θem_flw_max_sahz_i": 70,
            "θem_flw_min_sahz_i": 30,
        }
    ],

    "auxiliars_power": 0,
    "auxiliary_recovery_factor": 100,
    "emission_operation_time": 1,

    # ---- distribution ----
    "heat_losses_recovered": True,
    "distribution_loss_recovery": 90,
    "simplified_approach": 80,
    "distribution_aux_recovery": 80,
    "distribution_aux_power": 60,
    "distribution_loss_coeff": 60,
    "distribution_operation_time": 1,
    "recoverable_losses": 0.0,

    # ---- generation ----
    "full_load_power": 24,  # kW
    "max_monthly_load_factor": 100,
    "tH_gen_i_ON": 1,
    "auxiliary_power_generator": 0,
    "fraction_of_auxiliary_power_generator": 40,
    "generator_circuit": "direct",

    "gen_flow_temp_control_type": "Type A - Based on outdoor temperature",
    "gen_outdoor_temp_data": [
        {"θext_min_gen": -7, "θext_max_gen": 15, "θflw_gen_max": 70, "θflw_gen_min": 35}
    ],
    "θHW_gen_flw_const": 50.0,
    "speed_control_generator_pump": "variable",
    "generator_nominal_deltaT": 20,
    "efficiency_model": "simple",

    # ---- calc policy ----
    "calc_when_QH_positive_only": False,
    "off_compute_mode": "full",
}

# -----------------------------------------------------------------------------
# Required by main.py imports
# -----------------------------------------------------------------------------
BUILDING_ARCHETYPES: List[Dict[str, Any]] = [
    {
        "name": "SFH_Greece_1946_1969",
        "category": "Single Family House",
        "country": "Greece",
        "bui": BUI_SINGLE_FAMILY_1946_1969,
        "system": SYSTEM_CONDENSING_BOILER_RADIATORS,
    },
    {
        "name": "SFH_Italy_1946_1969",
        "category": "Single Family House",
        "country": "Italy",
        "bui": BUI_SINGLE_FAMILY_1946_1969,  # you can swap with an Italy-specific BUI later
        "system": SYSTEM_CONDENSING_BOILER_RADIATORS,
    },
]

# Used by /building endpoint (example only)
UNI11300_SIMULATION_EXAMPLE: Dict[str, Any] = {
    "hourly_sim": [
        {"timestamp": "2024-01-01 00:00:00", "Q_H": 2103.85, "Q_C": 0.0},
        {"timestamp": "2024-01-01 01:00:00", "Q_H": 1800.12, "Q_C": 0.0},
    ],
    "input_unit": "Wh",
    "heating_params": {
        "eta_emission": 0.95,
        "eta_distribution": 0.93,
        "eta_storage": 0.98,
        "eta_generation": 0.92,
        "f_recov_emission": 1.00,
        "f_recov_distribution": 0.80,
        "f_recov_storage": 0.90,
        "aux_emission_fraction": 0.005,
        "aux_distribution_fraction": 0.010,
        "aux_storage_fraction": 0.001,
        "aux_generation_fraction": 0.015,
        "fp_thermal": 1.05,
        "fp_electric": 2.18,
    },
    "cooling_params": {
        "eta_emission": 0.97,
        "eta_distribution": 0.95,
        "eta_storage": 1.00,
        "cop_generation": 3.2,
        "f_recov_emission": 1.00,
        "f_recov_distribution": 0.80,
        "f_recov_storage": 0.90,
        "aux_emission_fraction": 0.005,
        "aux_distribution_fraction": 0.010,
        "aux_storage_fraction": 0.001,
        "aux_generation_fraction": 0.005,
        "fp_electric": 2.18,
    },
}
