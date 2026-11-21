from __future__ import annotations

from typing import Any

from .spec import ArchetypeSpec, BuildingTypeClass, Country
from .thermal_mass import ThermalCapacityUnit, with_provisional_thermal_mass

# -----------------------------------------------------------------------------
# Envelope archetypes (provided dataset)
# -----------------------------------------------------------------------------
# Residential usage profiles, as in the standard tables of pybuildingenergy
# (EN 16798-1, Residential_detached_house / Residential_apartment).
_OCC_WEEKDAY_DEFAULT = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.5,
    0.5,
    0.5,
    0.1,
    0.1,
    0.1,
    0.2,
    0.2,
    0.2,
    0.2,
    0.5,
    0.5,
    0.5,
    0.8,
    0.8,
    0.8,
    1.0,
    1.0,
]
_OCC_WEEKEND_DEFAULT = [1.0] * 6 + [0.8] * 16 + [1.0] * 2
_APP_WEEKDAY_DEFAULT = [
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.7,
    0.7,
    0.5,
    0.5,
    0.6,
    0.6,
    0.6,
    0.6,
    0.5,
    0.5,
    0.7,
    0.7,
    0.8,
    0.8,
    0.8,
    0.6,
    0.6,
]
_APP_WEEKEND_DEFAULT = list(_APP_WEEKDAY_DEFAULT)
_LIGHT_DEFAULT = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.15,
    0.15,
    0.15,
    0.15,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
    0.15,
    0.15,
]
# Heating: 05:00-23:00 at setpoint, night setback (see temperature_setpoints).
_HEATING_PROFILE_DEFAULT = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
# Cooling follows occupancy: when people are at home (night/morning and evening
# on weekdays, whole day at the weekend); setpoint control does the rest.
_COOLING_PROFILE_WEEKDAY = [1] * 9 + [0] * 8 + [1] * 7
_COOLING_PROFILE_WEEKEND = [1] * 24
# Infiltration is continuous: it does not follow an on/off schedule.
_VENTILATION_PROFILE_DEFAULT = [1] * 24

# Total outdoor-air exchange of naturally ventilated dwellings (infiltration +
# window airing), by construction period. The oldest stock is leakiest; the
# newest tends towards the EN 16798-1 minimum (0.5 l/(s m2) ~ 0.67 1/h at 2.7 m).
_ACH_BY_PERIOD = {
    "0-1945": 0.9,
    "1946-1969": 0.8,
    "1970-1979": 0.75,
    "1980-1989": 0.7,
    "1990-1999": 0.65,
    "2000-2010": 0.6,
    "2011-now": 0.55,
}
# Linear thermal bridges as a delta-U on the gross envelope (W/m2K): worse in
# the pre-insulation stock, better in recent buildings (EN ISO 14683 ranges
# 0.10 / 0.05 / 0.02-0.03 for unaddressed / reduced / well-designed details).
_TB_DELTA_U_BY_PERIOD = {
    "0-1945": 0.10,
    "1946-1969": 0.10,
    "1970-1979": 0.08,
    "1980-1989": 0.08,
    "1990-1999": 0.06,
    "2000-2010": 0.05,
    "2011-now": 0.03,
}
# Full-load internal gains (W/m2) from the EN 16798-1 residential classes.
_OCCUPANT_LOAD_BY_CLASS = {
    BuildingTypeClass.DETACHED_HOUSE.value: 2.8,
    BuildingTypeClass.APARTMENT.value: 4.2,
}
_APPLIANCE_LOAD_BY_CLASS = {
    BuildingTypeClass.DETACHED_HOUSE.value: 2.4,
    BuildingTypeClass.APARTMENT.value: 3.0,
}


# Heating degree days per country: Eurostat nrg_chdd_a (indic_nrg=HDD), mean of 2014-2023.
# Used only to estimate the design outdoor temperature when sizing heating capacities.
# (The workbooks list 1415 for Belgium, which is wrong: Eurostat gives ~2541.)
_HDD_BY_COUNTRY = {
    Country.AUSTRIA.value: 3374,
    Country.BELGIUM.value: 2541,
    Country.BULGARIA.value: 2399,
    Country.CROATIA.value: 2182,
    Country.CYPRUS.value: 669,
    Country.CZECH_REPUBLIC.value: 3145,
    Country.DENMARK.value: 3059,
    Country.ESTONIA.value: 4032,
    Country.FINLAND.value: 5328,
    Country.FRANCE.value: 2229,
    Country.GERMANY.value: 2849,
    Country.GREECE.value: 1559,
    Country.HUNGARY.value: 2548,
    Country.IRELAND.value: 2694,
    Country.ITALY.value: 1873,
    Country.LATVIA.value: 3850,
    Country.LITHUANIA.value: 3652,
    Country.LUXEMBOURG.value: 2888,
    Country.MALTA.value: 466,
    Country.NETHERLANDS.value: 2499,
    Country.POLAND.value: 3156,
    Country.PORTUGAL.value: 1143,
    Country.ROMANIA.value: 2790,
    Country.SLOVAKIA.value: 3081,
    Country.SLOVENIA.value: 2657,
    Country.SPAIN.value: 1678,
    Country.SWEDEN.value: 5034,
}
_SOLAR_ABSORPTANCE_EXTERNAL = 0.6  # medium-coloured roof and walls, the same on every orientation
_COOLING_CAPACITY_W_PER_M2 = (
    100.0  # residential design cooling load, W per m2 of net floor area (hot climates peak ~100)
)


def _design_heating_capacity_w(
    *, ua_envelope: float, h_tb: float, ach: float, volume: float, hdd: float
) -> float:
    """Heating capacity (W) sized on the design heat loss of the building.

    (H_tr + H_tb + H_ve) x (20 C - theta_e,d) x 1.25, where theta_e,d is estimated
    from the country's degree days (7 - 0.0055 HDD, never above 0 C) and 1.25 covers
    the recovery from the night setback. Rounded up to 100 W.
    """
    h_ve = 0.34 * ach * volume
    theta_e_d = min(7.0 - 0.0055 * hdd, 0.0)
    watts = 1.25 * (ua_envelope + h_tb + h_ve) * (20.0 - theta_e_d)
    return float(-(-watts // 100) * 100)


def _constant_ach_ventilation(ach: float) -> dict[str, Any]:
    """Ventilation block read by pybuildingenergy >= 2.1.

    Without ``building_parameters.ventilation`` the library applies no
    ventilation/infiltration at all (``ventilation_type='none'``, H_ve = 0), and
    ``airflow_rates`` is ignored. The air-change rate is turned into a flow with
    ``building.volume`` (m3).
    """
    return {
        "components": [
            {
                "name": "infiltration",
                "ventilation_type": "constant_ach",
                "air_changes_per_hour": ach,
            }
        ],
        "units": "ACH (air changes per hour), volume in m3",
    }


# Default UNI/TS 11300 input for /run/iso52016-uni11300-pv.
UNI11300_SIMULATION_EXAMPLE: dict[str, Any] = {
    "input_unit": "Wh",
    "heating_params": {
        "eta_emission": 0.95,
        "eta_distribution": 0.93,
        "eta_storage": 0.98,
        "eta_generation": 0.80,
        "f_recov_emission": 1.00,
        "f_recov_distribution": 0.80,
        "f_recov_storage": 0.90,
        "aux_emission_fraction": 0.005,
        "aux_distribution_fraction": 0.010,
        "aux_storage_fraction": 0.001,
        "aux_generation_fraction": 0.015,
        "fp_thermal": 1.05,
        "fp_electric": 2.42,
    },
    "cooling_params": {
        "eta_emission": 0.97,
        "eta_distribution": 0.95,
        "eta_storage": 1.00,
        "cop_generation": 2.5,
        "f_recov_emission": 1.00,
        "f_recov_distribution": 0.80,
        "f_recov_storage": 0.90,
        "aux_emission_fraction": 0.005,
        "aux_distribution_fraction": 0.010,
        "aux_storage_fraction": 0.001,
        "aux_generation_fraction": 0.005,
        "fp_electric": 2.42,
    },
}

# -----------------------------------------------------------------------------
# Extended country archetypes provided by user
# -----------------------------------------------------------------------------
_FACADES = (("north", 0), ("east", 90), ("south", 180), ("west", 270))


def _make_extended_building(
    *,
    name: str,
    latitude: float,
    longitude: float,
    exposed_perimeter: float,
    wall_thickness: float,
    n_floors: float,
    building_type_class: str,
    net_floor_area: float,
    construction_class: str,
    roof_area: float,
    roof_u: float,
    roof_thermal_capacity: float,
    wall_area: float,
    wall_u: float,
    wall_thermal_capacity: float,
    slab_area: float,
    slab_u: float,
    slab_thermal_capacity: float,
    window_area: float,
    window_u: float,
    window_height: float,
    window_width: float,
    g_value: float = 0.76,
    thermal_bridge_delta_u: float = 0.08,
    infiltration_rate: float = 1.0,
    heating_capacity: float | None = None,
    cooling_capacity: float | None = None,
    hdd: float = 3000.0,
    occupants_full_load: float = 2.8,
    appliances_full_load: float = 2.4,
) -> dict[str, Any]:
    opaque_wall_area_per_side = (wall_area - window_area) / 4
    volume = slab_area * n_floors * 2.7
    h_tb = thermal_bridge_delta_u * (roof_area + wall_area + slab_area)
    if heating_capacity is None:
        ua_envelope = (
            roof_area * roof_u
            + (wall_area - window_area) * wall_u
            + window_area * window_u
            + 0.5 * slab_area * slab_u  # reduced ground heat loss
        )
        heating_capacity = _design_heating_capacity_w(
            ua_envelope=ua_envelope, h_tb=h_tb, ach=infiltration_rate, volume=volume, hdd=hdd
        )
    if cooling_capacity is None:
        cooling_capacity = _COOLING_CAPACITY_W_PER_M2 * net_floor_area
    window_area_per_side = window_area / 4
    window_width_per_side = window_width / 4

    return {
        "building": {
            "name": name,
            "azimuth_relative_to_true_north": 0.0,
            "latitude": latitude,
            "longitude": longitude,
            "exposed_perimeter": exposed_perimeter,
            "height": n_floors * 2.7,  # total building height, not one storey
            "wall_thickness": wall_thickness,
            "n_floors": n_floors,
            "building_type_class": building_type_class,
            "adj_zones_present": False,
            "number_adj_zone": 0,
            "net_floor_area": net_floor_area,
            # footprint x floors x storey height (2.7 m)
            "volume": volume,
            "construction_class": construction_class,
        },
        "building_surface": [
            {
                "name": "Roof surface",
                "type": "opaque",
                "area": roof_area,
                "sky_view_factor": 1.0,
                "u_value": roof_u,
                "solar_absorptance": _SOLAR_ABSORPTANCE_EXTERNAL,
                "thermal_capacity": roof_thermal_capacity,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None,
            },
            *(
                {
                    "name": f"Opaque {side} surface",
                    "type": "opaque",
                    "area": opaque_wall_area_per_side,
                    "sky_view_factor": 0.5,
                    "u_value": wall_u,
                    "solar_absorptance": _SOLAR_ABSORPTANCE_EXTERNAL,
                    "thermal_capacity": wall_thermal_capacity,
                    "orientation": {"azimuth": azimuth, "tilt": 90},
                    "name_adj_zone": None,
                }
                for side, azimuth in _FACADES
            ),
            {
                "name": "Slab to ground",
                "type": "opaque",
                "area": slab_area,
                "sky_view_factor": 0.0,
                "u_value": slab_u,
                "solar_absorptance": 0.6,
                "thermal_capacity": slab_thermal_capacity,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None,
            },
            *(
                {
                    "name": f"Transparent {side} surface",
                    "type": "transparent",
                    "area": window_area_per_side,
                    "sky_view_factor": 0.5,
                    "u_value": window_u,
                    "g_value": g_value,
                    "height": window_height,
                    "width": window_width_per_side,
                    "parapet": 1.0,
                    "orientation": {"azimuth": azimuth, "tilt": 90},
                    "shading": False,
                    "shading_type": "horizontal_overhang",
                    "width_or_distance_of_shading_elements": 0.5,
                    "overhang_proprieties": {"width_of_horizontal_overhangs": 0.8},
                    "name_adj_zone": None,
                }
                for side, azimuth in _FACADES
            ),
        ],
        "units": {
            "area": "m²",
            "u_value": "W/m²K",
            "thermal_capacity": ThermalCapacityUnit.TOTAL.value,
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
                "heating_capacity": heating_capacity,
                "cooling_capacity": cooling_capacity,
                "units": "W",
            },
            "airflow_rates": {
                "infiltration_rate": infiltration_rate,
                "units": "ACH (air changes per hour)",
            },
            "ventilation": _constant_ach_ventilation(infiltration_rate),
            "internal_gains": [
                {
                    "name": "occupants",
                    "full_load": occupants_full_load,
                    "weekday": _OCC_WEEKDAY_DEFAULT,
                    "weekend": _OCC_WEEKEND_DEFAULT,
                },
                {
                    "name": "appliances",
                    "full_load": appliances_full_load,
                    "weekday": _APP_WEEKDAY_DEFAULT,
                    "weekend": _APP_WEEKEND_DEFAULT,
                },
                {
                    "name": "lighting",
                    "full_load": 3.0,
                    "weekday": _LIGHT_DEFAULT,
                    "weekend": _LIGHT_DEFAULT,
                },
            ],
            "construction": {
                "wall_thickness": wall_thickness,
                # pybuildingenergy >= 2.1 reads this as a total H_tb in W/K:
                # delta-U (W/m2K) x gross envelope area (roof + walls + slab).
                "thermal_bridge_heat_W_K": h_tb,
                "units": "m (thickness), W/K (thermal bridges)",
            },
            "climate_parameters": {"coldest_month": 1, "units": "1-12 (January-December)"},
            "heating_profile": {
                "weekday": _HEATING_PROFILE_DEFAULT,
                "weekend": _HEATING_PROFILE_DEFAULT,
            },
            "cooling_profile": {
                "weekday": _COOLING_PROFILE_WEEKDAY,
                "weekend": _COOLING_PROFILE_WEEKEND,
            },
            "ventilation_profile": {
                "weekday": _VENTILATION_PROFILE_DEFAULT,
                "weekend": _VENTILATION_PROFILE_DEFAULT,
            },
        },
    }


def _make_extended_uni11300(
    *,
    heating_eta_generation: float,
    cooling_cop_generation: float,
    dhw_eta_generation: float,
    ventilation_rate_1_per_h: float = 0.8,
    fp_electric: float = 2.18,
) -> dict[str, Any]:
    return {
        "input_unit": "Wh",
        "heating_params": {
            "eta_emission": 0.95,
            "eta_distribution": 0.93,
            "eta_storage": 0.98,
            "eta_generation": heating_eta_generation,
            "f_recov_emission": 1.0,
            "f_recov_distribution": 0.8,
            "f_recov_storage": 0.9,
            "aux_emission_fraction": 0.005,
            "aux_distribution_fraction": 0.01,
            "aux_storage_fraction": 0.001,
            "aux_generation_fraction": 0.015,
            "fp_thermal": 1.05,
            "fp_electric": fp_electric,
        },
        "cooling_params": {
            "eta_emission": 0.97,
            "eta_distribution": 0.98,
            "eta_storage": 1.0,
            "cop_generation": cooling_cop_generation,
            "f_recov_emission": 1.0,
            "f_recov_distribution": 0.8,
            "f_recov_storage": 0.9,
            "aux_emission_fraction": 0.005,
            "aux_distribution_fraction": 0.01,
            "aux_storage_fraction": 0.001,
            "aux_generation_fraction": 0.005,
            "fp_electric": fp_electric,
        },
        "dhw_params": {
            "eta_generation": dhw_eta_generation,
            "eta_distribution": 0.9,
            "eta_storage": 0.9,
            "aux_fraction": 0.01,
            "fp_thermal": 1.05,
            "fp_electric": fp_electric,
        },
        "ventilation_params": {
            "design_air_change_rate_1_per_h": ventilation_rate_1_per_h,
            "heat_recovery": False,
        },
    }


# SYSTEM: JSON-safe (condensing boiler + radiators)
# NOTE: fields that were DataFrames are now list-of-dict.
# -----------------------------------------------------------------------------
SYSTEM_CONDENSING_BOILER_RADIATORS: dict[str, Any] = {
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
        {
            "θext_min_gen": -7,
            "θext_max_gen": 15,
            "θflw_gen_max": 70,
            "θflw_gen_min": 35,
        }
    ],
    "θHW_gen_flw_const": 50.0,
    "speed_control_generator_pump": "variable",
    "generator_nominal_deltaT": 20,
    "efficiency_model": "simple",
    # ---- calc policy ----
    "calc_when_QH_positive_only": False,
    "off_compute_mode": "full",
}


@with_provisional_thermal_mass
def build_archetype(name: str, spec: ArchetypeSpec) -> dict[str, Any]:
    # Some specs ship a placeholder net_floor_area of 0.0. A zero area breaks
    # per-m2 metrics and gets silently coerced to 1.0 m2 by
    # sanitize_and_validate_BUI(fix=True), which distorts internal gains,
    # thermal capacity, and occupancy ventilation in the ISO 52016 run.
    # Derive it from the footprint instead: ground slab area x floors.
    # (The Belgian workbook gives the area per *dwelling*, while footprint, walls
    # and perimeter are per *building*, so those specs also use the placeholder.)
    net_floor_area = spec["net_floor_area"]
    if net_floor_area <= 0:
        net_floor_area = spec["slab_area"] * spec["n_floors"]
    period = name.split("_", 2)[2]  # e.g. 'AT_SFH_1946-1969' -> '1946-1969'
    bui = _make_extended_building(
        name=name,
        latitude=spec["latitude"],
        longitude=spec["longitude"],
        exposed_perimeter=spec["exposed_perimeter"],
        wall_thickness=spec["wall_thickness"],
        n_floors=spec["n_floors"],
        building_type_class=spec["building_type_class"],
        net_floor_area=net_floor_area,
        construction_class=spec["construction_class"],
        roof_area=spec["roof_area"],
        roof_u=spec["roof_u"],
        roof_thermal_capacity=spec["roof_thermal_capacity"],
        wall_area=spec["wall_area"],
        wall_u=spec["wall_u"],
        wall_thermal_capacity=spec["wall_thermal_capacity"],
        slab_area=spec["slab_area"],
        slab_u=spec["slab_u"],
        slab_thermal_capacity=spec["slab_thermal_capacity"],
        window_area=spec["window_area"],
        window_u=spec["window_u"],
        window_height=spec["window_height"],
        window_width=spec["window_width"],
        g_value=spec.get("g_value", 0.76),
        hdd=_HDD_BY_COUNTRY[spec["country"]],
        thermal_bridge_delta_u=spec.get("thermal_bridge_delta_u", _TB_DELTA_U_BY_PERIOD[period]),
        infiltration_rate=spec.get("infiltration_rate", _ACH_BY_PERIOD[period]),
        occupants_full_load=spec.get(
            "occupants_full_load", _OCCUPANT_LOAD_BY_CLASS[spec["building_type_class"]]
        ),
        appliances_full_load=spec.get(
            "appliances_full_load", _APPLIANCE_LOAD_BY_CLASS[spec["building_type_class"]]
        ),
    )
    uni = _make_extended_uni11300(
        heating_eta_generation=spec["heating_eta_generation"],
        cooling_cop_generation=spec["cooling_cop_generation"],
        dhw_eta_generation=spec["dhw_eta_generation"],
        ventilation_rate_1_per_h=spec.get("ventilation_rate_1_per_h", _ACH_BY_PERIOD[period]),
        fp_electric=spec.get("fp_electric", 2.18),
    )
    return {
        "name": name,
        "category": spec["category"],
        "country": spec["country"],
        "bui": bui,
        "system": SYSTEM_CONDENSING_BOILER_RADIATORS,
        "systems_archetype": uni,
        "uni11300": uni,
    }
