import numpy as np
import pandas as pd
from typing import List, Dict, Any

# DATABSE OF ARCHETPYE

# Greece
BUI_SINGLE_FAMILY_1946_1969 = {
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
        "number_adj_zone":0,
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
            "orientation": {
                "azimuth": 0,
                "tilt": 0
            },
            "name_adj_zone": None
        },
        {
            "name": "Opaque north surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 0.95,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 0,
                "tilt": 90
            },
            "name_adj_zone": None
        },
        {
            "name": "Opaque south surface",
            # "type": "opaque",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 0.95,
            "solar_absorptance": 0.4,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 180,
                "tilt": 90
            },
            "name_adj_zone": None
        },
        {
            "name": "Opaque east surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 0.95,
            "solar_absorptance": 0.6,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 90,
                "tilt": 90
            },
            "name_adj_zone": None
        },
        {
            "name": "Opaque west surface",
            "type": "opaque",
            "area": 30,
            "sky_view_factor": 0.5,
            "u_value": 0.95,
            "solar_absorptance": 0.7,
            "thermal_capacity": 1416240.0,
            "orientation": {
                "azimuth": 270,
                "tilt": 90
            },
            "name_adj_zone": None
        },
        {
            "name": "Slab to ground",
            "type": "opaque",
            "area": 120,
            "sky_view_factor": 0.0,
            "u_value": 1.2,
            "solar_absorptance": 0.6,
            "thermal_capacity": 405801,
            "orientation": {
                "azimuth": 0,
                "tilt": 0
            },
            "name_adj_zone": None
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
            "orientation": {
                "azimuth": 90,
                "tilt": 90
            },
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {
                "width_of_horizontal_overhangs":1
            },
            "name_adj_zone": None
        },
        {
            "name": "Transparent west surface",
            "type": "transparent",
            "area": 4,
            "sky_view_factor": 0.5,
            "u_value": 5,
            "g_value": 0.726,
            "height": 2,
            "width": 1,
            "parapet": 1.1,
            "orientation": {
                "azimuth": 270,
                "tilt": 90
            },
            "shading": False,
            "shading_type": "horizontal_overhang",
            "width_or_distance_of_shading_elements": 0.5,
            "overhang_proprieties": {
                "width_of_horizontal_overhangs":1
            },
            "name_adj_zone": None
        }
    ],
    "units": {
        "area": "m²",
        "u_value": "W/m²K",
        "thermal_capacity": "J/kgK",
        "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
        "tilt": "degrees (0=horizontal, 90=vertical)",
        "internal_gain": "W/m²",
        "internal_gain_profile": "Normalized to 0-1",
        "HVAC_profile": "0: off, 1: on"
    },
    "building_parameters": {
        "temperature_setpoints": {
            "heating_setpoint": 20.0,
            "heating_setback": 17.0,
            "cooling_setpoint": 26.0,
            "cooling_setback": 30.0,
            "units": "°C"
        },
        "system_capacities": {
            "heating_capacity": 10000000.0,
            "cooling_capacity": 12000000.0,
            "units": "W"
        },
        "airflow_rates": {
            "infiltration_rate": 1.0,
            # "ventilation_rate_extra": 1.0,
            "units": "ACH (air changes per hour)"
        },
        "internal_gains": [
            {
                "name": "occupants",
                "full_load": 4.2,
                "weekday": [1.0,1.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.5,0.5,0.5,0.8,0.8,0.8,1.0,1.0],
                "weekend": [1.0,1.0,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,1.0,1.0]
            },
            {
                "name": "appliances",
                "full_load": 3,
                "weekday": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
                "weekend": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
            },
            {
                "name": "lighting",
                "full_load": 3,
                "weekday": [0.0,0.0,0.0,0.0,0.0,0.0,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2,0.15,0.15],
                "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2,0.15,0.15],
            }
        ],
        "construction": {
            "wall_thickness": 0.3,
            "thermal_bridges": 0.8,
            "units": "m (for thickness), W/mK (for thermal bridges)"
        },
        "climate_parameters": {
            "coldest_month": 1,
            "units": "1-12 (January-December)"
        },
        "heating_profile": {
            "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
            "weekend": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
        },
        "cooling_profile": {
            "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
            "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0]
        },
        "ventilation_profile": {
            "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
            "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0]
        }
    }
}



'''
CONDENSING BOILER + RADIATORS
'''
INPUT_SYSTEM_HVAC_CONDENSING_BOILER_AND_RADIATOR = {
    # ======================
    # ---- EMISSION (secondary side) ----
    # ======================
    'emitter_type': 'Radiator',          # Must match a TB14 index (e.g., 'Radiator' / 'Floor heating' / 'Fan coil')
    'nominal_power': 10,                 # kW: nominal emitter/zone power at design conditions
    'emission_efficiency': 95,           # %: emission-side efficiency (accounts for emission-side losses)
    'selected_emm_cont_circuit': 1,      # 0=C.2, 1=C.3, 2=C.4, 3=C.5; radiators are typically modeled as C.3 (variable flow)
    'flow_temp_control_type': 'Type 2 - Based on outdoor temperature',  # Secondary control law (if used): weather-compensated curve
    'mixing_valve': False,               # Radiator circuits often do not have a dedicated mixing valve on the secondary side
    'mixing_valve_delta': 0,             # °C: mixing valve offset; keep 0 if mixing_valve=False for consistency

    # Secondary-side emission limits/setpoints for radiators
    # Typical medium/low-temperature values to encourage condensing operation (e.g., around 55/45), with an upper cap (e.g., 70°C)
    'heat_emission_data': pd.DataFrame({
        "θH_em_flw_max_sahz_i": [70],     # °C: maximum allowed radiator supply temperature
        "ΔθH_em_w_max_sahz_i":  [20],     # K : maximum allowed supply/return temperature difference
        "θH_em_ret_req_sahz_i": [45],     # °C: desired return temperature (lower return improves condensing)
        "βH_em_req_sahz_i":     [80],     # %: required duty-cycle/load factor (mainly relevant for C.4/C.5 ON-OFF behavior)
        "θH_em_flw_min_tz_i":   [30],     # °C: minimum allowed radiator supply temperature
    }, index=[
        "Max flow temperature HZ1",
        "Max Δθ flow / return HZ1",
        "Desired return temperature HZ1",
        "Desired load factor with ON-OFF for HZ1",
        "Minimum flow temperature for HZ1"
    ]),

    # Secondary weather-compensation curve (used if flow_temp_control_type = Type 2)
    'outdoor_temp_data': pd.DataFrame({
        "θext_min_sahz_i":     [-10],     # °C: minimum outdoor temperature for the curve
        "θext_max_sahz_i":     [16],      # °C: maximum outdoor temperature for the curve
        "θem_flw_max_sahz_i":  [70],      # °C: secondary supply at minimum outdoor temperature
        "θem_flw_min_sahz_i":  [30],      # °C: secondary supply at maximum outdoor temperature
    }, index=[
        "Minimum outdoor temperature",
        "Maximum outdoor temperature",
        "Maximum flow temperature",
        "Minimum flow temperature"
    ]),

    # Emission-side auxiliaries (only if you want to account for them)
    'auxiliars_power': 0,                # W : e.g., zone actuators/valves/secondary-side pumps (if any)
    'auxiliary_recovery_factor': 100,    # %: recoverable share of auxiliary energy (if modeled as recoverable heat)
    'emission_operation_time': 1,        # h : timestep duration (typically 1 hour for hourly inputs)

    # ======================
    # ---- DISTRIBUTION ----
    # ======================
    'heat_losses_recovered': True,       # If True, part of distribution losses can be treated as recoverable heat
    'distribution_loss_recovery': 90,    # %: recoverable fraction of distribution heat losses (towards conditioned zones)
    'simplified_approach': 80,           # %: simplified/holistic approach factor used by your implementation
    'distribution_aux_recovery': 80,     # %: recoverable fraction of distribution auxiliaries (pump heat gains)
    'distribution_aux_power': 60,        # W : distribution pump electric power (typical order: 40–100 W)
    'distribution_loss_coeff': 60,       # W/K: distribution heat loss coefficient (depends on pipe length/insulation)
    'distribution_operation_time': 1,    # h : timestep duration for distribution calculations
    'recoverable_losses': 0.0,           # kWh: additional system-level recoverable losses upstream (if any)

    # ======================
    # ---- GENERATION (primary side): condensing boiler ----
    # ======================
    'full_load_power': 24,               # kW: boiler nominal useful power
    'max_monthly_load_factor': 100,      # % : cap on delivered energy/power (per your implementation)
    'tH_gen_i_ON': 1,                    # h : timestep duration for generator calculations

    # Generator auxiliaries (if you model auxiliary electric use as a % of EHW_gen_in)
    'auxiliary_power_generator': 0,      # % of EHW_gen_in (set 0 if you do not want this auxiliary model)
    'fraction_of_auxiliary_power_generator': 40,  # %: recoverable share (if interpreted as recoverable thermal losses)

    # Hydraulic layout on the generator side
    'generator_circuit': 'direct',       # 'direct' means primary flow equals secondary flow (no hydraulic separation)

    # PRIMARY supply temperature control
    'gen_flow_temp_control_type': 'Type A - Based on outdoor temperature',  # Primary weather-compensated curve
    'gen_outdoor_temp_data': pd.DataFrame({
        "θext_min_gen": [-7],            # °C: minimum outdoor temperature for the primary curve
        "θext_max_gen": [15],            # °C: maximum outdoor temperature for the primary curve
        "θflw_gen_max": [70],            # °C: maximum primary supply temperature
        "θflw_gen_min": [35],            # °C: minimum primary supply temperature
    }, index=["Generator curve"]),
    'θHW_gen_flw_const': 50.0,           # °C: used only if gen_flow_temp_control_type is Type C (constant)

    # Generator pump control
    'speed_control_generator_pump': 'variable',  # 'deltaT_constant' | 'variable'
    'generator_nominal_deltaT': 20,              # K : typical radiator ΔT (e.g., 70/50 or 55/35)

    # Generator efficiency model (condensing behavior proxy)
    'efficiency_model': 'simple',        # Starting point; you may switch to 'parametric' for more explicit control
    # Optional parameters if you switch to 'parametric'
    # 'eta_max': 108.0,                  # %: max efficiency in strong condensing conditions
    # 'eta_no_cond': 94.0,               # %: efficiency without condensing
    # 'T_ret_min': 25.0,                 # °C: reference minimum return temperature for max efficiency
    # 'T_ret_thr': 55.0,                 # °C: threshold return temperature above which condensing benefit vanishes

    # ======================
    # ---- CALCULATION POLICY ----
    # ======================
    'calc_when_QH_positive_only': False,  # If False: always compute (including zero/negative load timesteps)
    'off_compute_mode': 'full',           # If calc_when_QH_positive_only=True: 'idle' | 'temps' | 'full'
}

# ====================================================================================================================


INPUT_SYSTEM_HVAC = {
    # ---- emitter ----
    'emitter_type': 'Floor heating',
    'nominal_power': 8,
    'emission_efficiency': 90,
    'flow_temp_control_type': 'Type 2 - Based on outdoor temperature',
    'selected_emm_cont_circuit': 0,
    'mixing_valve': True,
    # 'TB14': custom_TB14, #  <- Uncomment and upload your emittere table, oterwhise the default stored in gloabl_inputs.py is used
    # 'heat_emission_data' : pd.DataFrame({ <- Uncomment and upload your emittere table, oterwhise the default stored in gloabl_inputs.py is used
    #         "θH_em_flw_max_sahz_i": [45],
    #         "ΔθH_em_w_max_sahz_i": [8],
    #         "θH_em_ret_req_sahz_i": [20],
    #         "βH_em_req_sahz_i": [80],
    #         "θH_em_flw_min_tz_i": [28],
    #     }, index=[
    #         "Max flow temperature HZ1",
    #         "Max Δθ flow / return HZ1",
    #         "Desired return temperature HZ1",
    #         "Desired load factor with ON-OFF for HZ1",
    #         "Minimum flow temperature for HZ1"
    #     ]),
    'mixing_valve_delta':2,
    # 'constant_flow_temp':42,

    # --- distribution ---
    'heat_losses_recovered': True,
    'distribution_loss_recovery': 90,
    'simplified_approach': 80,
    'distribution_aux_recovery': 80,
    'distribution_aux_power': 30,
    'distribution_loss_coeff': 48,
    'distribution_operation_time': 1,
    
    # --- generator ---
    'full_load_power': 27,                  # kW
    'max_monthly_load_factor': 100,         # %
    'tH_gen_i_ON': 1,                       # h
    'auxiliary_power_generator': 0,         # %
    'fraction_of_auxiliary_power_generator': 40,   # %
    'generator_circuit': 'independent',     # 'direct' | 'independent'

    # Primary: independent climatic curve
    'gen_flow_temp_control_type': 'Type A - Based on outdoor temperature',
    'gen_outdoor_temp_data': pd.DataFrame({
        "θext_min_gen": [-7],
        "θext_max_gen": [15],
        "θflw_gen_max": [60],
        "θflw_gen_min": [35],
    }, index=["Generator curve"]),

    'speed_control_generator_pump': 'variable',
    'generator_nominal_deltaT': 20,         # °C
    'mixing_valve_delta':2,

    # Optional explicit generator setpoints (commented by default)
    # 'θHW_gen_flw_set': 50,
    # 'θHW_gen_ret_set': 40,

    # Efficiency model
    'efficiency_model': 'simple',

    # Calculation options
    'calc_when_QH_positive_only': False,
    'off_compute_mode': 'full',

}

# ==== EXAMPLE OF BUI TO PASS TO THE API ENDPOINT ====
'''
{
  "building": {
    "name": "test-cy",
    "azimuth_relative_to_true_north": 41.8,
    "latitude": 46.49018685497359,
    "longitude": 11.327028776009655,
    "exposed_perimeter": 40,
    "height": 3,
    "wall_thickness": 0.3,
    "n_floors": 1,
    "building_type_class": "Residential_apartment",
    "adj_zones_present": false,
    "number_adj_zone": 2,
    "net_floor_area": 100,
    "construction_class": "class_i"
  },

  "adjacent_zones": [
    {
      "name": "adj_1",
      "orientation_zone": { "azimuth": 0 },
      "area_facade_elements": [20, 60, 30, 30, 50, 50],
      "typology_elements": ["OP", "OP", "OP", "OP", "GR", "OP"],
      "transmittance_U_elements": [
        0.8196721311475411,
        0.8196721311475411,
        0.8196721311475411,
        0.8196721311475411,
        0.5156683855612851,
        1.162633192818565
      ],
      "orientation_elements": ["NV", "SV", "EV", "WV", "HOR", "HOR"],
      "volume": 300,
      "building_type_class": "Residential_apartment",
      "a_use": 50
    },
    {
      "name": "adj_2",
      "orientation_zone": { "azimuth": 180 },
      "area_facade_elements": [20, 60, 30, 30, 50, 50],
      "typology_elements": ["OP", "OP", "OP", "OP", "GR", "OP"],
      "transmittance_U_elements": [
        0.8196721311475411,
        0.8196721311475411,
        0.8196721311475411,
        0.8196721311475411,
        0.5156683855612851,
        1.162633192818565
      ],
      "orientation_elements": ["NV", "SV", "EV", "WV", "HOR", "HOR"],
      "volume": 300,
      "building_type_class": "Residential_apartment",
      "a_use": 50
    }
  ],

  "building_surface": [
    {
      "name": "Roof surface",
      "type": "opaque",
      "area": 130,
      "sky_view_factor": 1.0,
      "u_value": 2.2,
      "solar_absorptance": 0.4,
      "thermal_capacity": 741500.0,
      "orientation": { "azimuth": 0, "tilt": 0 },
      "name_adj_zone": null
    },
    {
      "name": "Opaque north surface",
      "type": "opaque",
      "area": 30,
      "sky_view_factor": 0.5,
      "u_value": 1.4,
      "solar_absorptance": 0.4,
      "thermal_capacity": 1416240.0,
      "orientation": { "azimuth": 0, "tilt": 90 },
      "name_adj_zone": "adj_1"
    },
    {
      "name": "Opaque south surface",
      "type": "opaque",
      "area": 30,
      "sky_view_factor": 0.5,
      "u_value": 1.4,
      "solar_absorptance": 0.4,
      "thermal_capacity": 1416240.0,
      "orientation": { "azimuth": 180, "tilt": 90 },
      "name_adj_zone": "adj_2"
    },
    {
      "name": "Opaque east surface",
      "type": "opaque",
      "area": 30,
      "sky_view_factor": 0.5,
      "u_value": 1.2,
      "solar_absorptance": 0.6,
      "thermal_capacity": 1416240.0,
      "orientation": { "azimuth": 90, "tilt": 90 },
      "name_adj_zone": null
    },
    {
      "name": "Opaque west surface",
      "type": "opaque",
      "area": 30,
      "sky_view_factor": 0.5,
      "u_value": 1.2,
      "solar_absorptance": 0.7,
      "thermal_capacity": 1416240.0,
      "orientation": { "azimuth": 270, "tilt": 90 },
      "name_adj_zone": null
    },
    {
      "name": "Slab to ground",
      "type": "opaque",
      "area": 100,
      "sky_view_factor": 0.0,
      "u_value": 1.6,
      "solar_absorptance": 0.6,
      "thermal_capacity": 405801,
      "orientation": { "azimuth": 0, "tilt": 0 },
      "name_adj_zone": null
    },
    {
      "name": "Transparent east surface",
      "type": "transparent",
      "area": 4,
      "sky_view_factor": 0.5,
      "u_value": 5,
      "g_value": 0.726,
      "height": 2,
      "width": 1,
      "parapet": 1.1,
      "orientation": { "azimuth": 90, "tilt": 90 },
      "shading": false,
      "shading_type": "horizontal_overhang",
      "width_or_distance_of_shading_elements": 0.5,
      "overhang_proprieties": {
        "width_of_horizontal_overhangs": 1
      },
      "name_adj_zone": null
    },
    {
      "name": "Transparent west surface",
      "type": "transparent",
      "area": 4,
      "sky_view_factor": 0.5,
      "u_value": 5,
      "g_value": 0.726,
      "height": 2,
      "width": 1,
      "parapet": 1.1,
      "orientation": { "azimuth": 270, "tilt": 90 },
      "shading": false,
      "shading_type": "horizontal_overhang",
      "width_or_distance_of_shading_elements": 0.5,
      "overhang_proprieties": {
        "width_of_horizontal_overhangs": 1
      },
      "name_adj_zone": null
    }
  ],

  "units": {
    "area": "m²",
    "u_value": "W/m²K",
    "thermal_capacity": "J/kgK",
    "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
    "tilt": "degrees (0=horizontal, 90=vertical)",
    "internal_gain": "W/m²",
    "internal_gain_profile": "Normalized to 0-1",
    "HVAC_profile": "0: off, 1: on"
  },

  "building_parameters": {
    "temperature_setpoints": {
      "heating_setpoint": 20.0,
      "heating_setback": 17.0,
      "cooling_setpoint": 26.0,
      "cooling_setback": 30.0,
      "units": "°C"
    },
    "system_capacities": {
      "heating_capacity": 10000000.0,
      "cooling_capacity": 12000000.0,
      "units": "W"
    },
    "airflow_rates": {
      "infiltration_rate": 1.0,
      "units": "ACH (air changes per hour)"
    },
    "internal_gains": [
      {
        "name": "occupants",
        "full_load": 4.2,
        "weekday": [1.0,1.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.5,0.5,0.5,0.8,0.8,0.8,1.0,1.0],
        "weekend": [1.0,1.0,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,1.0,1.0]
      },
      {
        "name": "appliances",
        "full_load": 3,
        "weekday": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6],
        "weekend": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.6,0.6,0.6,0.6,0.5,0.5,0.7,0.7,0.8,0.8,0.8,0.6,0.6]
      },
      {
        "name": "lighting",
        "full_load": 3,
        "weekday": [0.0,0.0,0.0,0.0,0.0,0.0,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2,0.15,0.15],
        "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.2,0.2,0.2,0.2,0.2,0.15,0.15]
      }
    ],
    "construction": {
      "wall_thickness": 0.3,
      "thermal_bridges": 2,
      "units": "m (for thickness), W/mK (for thermal bridges)"
    },
    "climate_parameters": {
      "coldest_month": 1,
      "units": "1-12 (January-December)"
    },
    "heating_profile": {
      "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
      "weekend": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0]
    },
    "cooling_profile": {
      "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
      "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0]
    },
    "ventilation_profile": {
      "weekday": [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
      "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0]
    }
  }
}
{
  "emitter_type": "Floor heating",
  "nominal_power": 8,
  "emission_efficiency": 90,
  "flow_temp_control_type": "Type 2 - Based on outdoor temperature",
  "selected_emm_cont_circuit": 0,
  "mixing_valve": true,
  "mixing_valve_delta": 2,

  "heat_losses_recovered": true,
  "distribution_loss_recovery": 90,
  "simplified_approach": 80,
  "distribution_aux_recovery": 80,
  "distribution_aux_power": 30,
  "distribution_loss_coeff": 48,
  "distribution_operation_time": 1,

  "full_load_power": 27,
  "max_monthly_load_factor": 100,
  "tH_gen_i_ON": 1,
  "auxiliary_power_generator": 0,
  "fraction_of_auxiliary_power_generator": 40,
  "generator_circuit": "independent",

  "gen_flow_temp_control_type": "Type A - Based on outdoor temperature",

  "gen_outdoor_temp_data": [
    {
      "θext_min_gen": -7,
      "θext_max_gen": 15,
      "θflw_gen_max": 60,
      "θflw_gen_min": 35
    }
  ],

  "speed_control_generator_pump": "variable",
  "generator_nominal_deltaT": 20,
  "efficiency_model": "simple",
  "calc_when_QH_positive_only": false,
  "off_compute_mode": "full"
}

'''


# Lista di edifici di esempio (archetipi)
# Puoi duplicare e cambiare category/country/name per coprire altri casi
BUILDING_ARCHETYPES: List[Dict[str, Any]] = [
    {
        "name": "SFH_Greece_1946_1969",
        "category": "Single Family House",
        "country": "Greece",
        "bui": BUI_SINGLE_FAMILY_1946_1969,
        "system": INPUT_SYSTEM_HVAC_CONDENSING_BOILER_AND_RADIATOR,
    },
    {
        "name": "SFH_Italy_1946_1969",
        "category": "Single Family House",
        "country": "Italy",
        "bui": BUI_SINGLE_FAMILY_1946_1969,
        "system": INPUT_SYSTEM_HVAC_CONDENSING_BOILER_AND_RADIATOR,
    },
]

# Example input payload for UNI/TS 11300 primary energy calculation
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
