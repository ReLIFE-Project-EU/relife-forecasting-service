# ReLIFE Forecasting Service - API README

API for a building thermal simulation using the iso 52000 (ISO 52016, ISO 52010, etc.) and 15316, based on the pybuildingenergy python library.

This API exposes endpoints to:

- retrieve building/system configurations (from archetypes or custom input)
- validate BUI (Building Unit Input) models and HVAC/system configurations
- simulate energy needs (ISO 52016) and systems (ISO 15316)
- run batch simulations in parallel
- generate HTML energy analysis reports
- update U-values for EPC Greece and simulate ECM scenarios (U-values)
- compute CO2e emissions and compare scenarios

> Note: the app also includes endpoints mounted via `health.router`.
> They are not shown in `main.py` below: refer to `relife_forecasting/routes/health.py`
> (or the Swagger docs) for details.

---

## Local Run using virtual environment

- **Activate virtual environment and install packages**

    ```bash
    pipenv shell 
    pipenv install 
    ```
    ** install al packeages define in pipfile

- **Run the uvicorn server (FastAPI)**

    ```bash
    uvicorn main:app --reload --port 9091
    ```
    ** tested with python 3.11 


---

## Conventions

### Base URL
Depends on deployment; locally it is typically:

- `http://127.0.0.1:9091`

### OpenAPI documentation (Swagger)

- `GET /docs` (Swagger UI)
- `GET /openapi.json` (OpenAPI schema)

### Content-Type
Endpoints accept:

- `application/json` for JSON bodies
- `multipart/form-data` for file uploads (EPW) and `Form(...)` fields

### Errors

- `400 Bad Request`: missing/invalid params or body
- `404 Not Found`: archetype/country not found, etc.
- `422 Unprocessable Entity`: BUI validation failed ("ERROR" issues)
- `500 Internal Server Error`: runtime errors (e.g., HTML report not generated)

---

## Archetypes and input modes

Many endpoints support two modes:

1. **archetype=true** (default): BUI and system are read from `BUILDING_ARCHETYPES` filtered by:
   - `category`
   - `country`
   - `name`

2. **archetype=false**: BUI and system come from the request:
   - some endpoints use `application/json` (`payload` with keys `bui` and `system`)
   - other endpoints use `multipart/form-data` (`bui_json` and `system_json` as JSON strings)

---

# Endpoints

## 1) POST `/building`
**Purpose**: returns a BUI + system configuration.

### Archetype mode (`archetype=true`)
**Query params (required)**:
- `archetype` (bool, default `true`)
- `category` (string, **required**)
- `country` (string, **required**)
- `name` (string, **required**)

**Body**: none.

### Custom mode (`archetype=false`)
**Query params**:
- `archetype=false`

**Body JSON (required)** (`application/json`):
```json
{
  "bui": { ... },
  "system": { ... }
}
```
Required fields: `bui`, `system`.

### Response (200)
```json
{
  "source": "archetype" | "custom",
  "name": "...",
  "category": "...",
  "country": "...",
  "bui": { ... },
  "system": { ... }
}
```

### Example building-HVAC system input

```bash 

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

```

---

## 2) GET `/building/available`
**Purpose**: lists available archetypes (metadata only, no full BUI).

### Input
None.

### Response (200)
```json
[
  {"name": "SFH_Italy_default", "category": "Single Family House", "country": "Italy"},
  ...
]
```

---

## 3) POST `/validate`
**Purpose**: validate BUI + system. Applies sanitization (fix=True) then validates (fix=False). If there are "ERROR" issues it returns 422.

### Archetype mode (`archetype=true`)
**Query params (required)**:
- `archetype` (bool, default `true`)
- `category` (string, **required**)
- `country` (string, **required**)
- `name` (string, **required**)

### Custom mode (`archetype=false`)
**Query params**:
- `archetype=false`

**Body JSON (required)**:
```json
{
  "bui": { ... },
  "system": { ... }
}
```

### Response (200)
Returns:
- `bui_fixed`: BUI with automatic fixes
- `bui_checked`: validated BUI
- `system_checked`: system configuration (possibly normalized)
- `bui_report_fixed`: sanitization report
- `bui_issues`: issue list (warning/error)
- `system_messages`: HVAC diagnostics

Example (schema):
```json
{
  "source": "archetype" | "custom",
  "name": "...",
  "category": "...",
  "country": "...",
  "bui_fixed": { ... },
  "bui_checked": { ... },
  "system_checked": { ... },
  "bui_report_fixed": { ... },
  "bui_issues": [ ... ],
  "system_messages": [ ... ]
}
```

### Response (422)
```json
{
  "detail": {
    "msg": "Errors in BUI model.",
    "errors": [ ... ],
    "system_messages": [ ... ]
  }
}
```

---

## 4) POST `/simulate`
**Purpose**: simulate a single building.

Pipeline:
1) input selection (archetype or custom)
2) validation (`validate_bui_and_system`)
3) ISO 52016 calculation (needs) using PVGIS weather or EPW file
4) ISO 15316 heating system calculation

### Common input (Query params)
- `archetype` (bool, default `true`)
- `weather_source` (string, default `pvgis`, values: `pvgis` | `epw`)

If `archetype=true`, these are **required**:
- `category`
- `country`
- `name`

### Archetype mode (`archetype=true`)
**Body**: none.

### Custom mode (`archetype=false`)
**Content-Type**: `multipart/form-data`

**Required fields**:
- `bui_json` (JSON string)
- `system_json` (JSON string)

### EPW weather
If `weather_source=epw` then **required**:
- `epw_file` (file upload)

### Response (200)
```json
{
  "source": "archetype" | "custom",
  "name": "...",
  "category": "...",
  "country": "...",
  "weather_source": "pvgis" | "epw",
  "validation": {
    "bui_issues": [ ... ],
    "system_messages": [ ... ]
  },
  "results": {
    "hourly_building": [ {"timestamp": "...", "Q_HC": ..., ...}, ... ]
  }
}
```
> Note: in code, `annual_building` and `hourly_system` are commented out to reduce response size.

---

## 5) POST `/simulate/batch`
**Purpose**: simulate multiple buildings in parallel (multiprocessing). Returns detailed results and a summary table.

### Body JSON (required)
Contains `mode`, which can be:

#### A) `mode="archetype"`
```json
{
  "mode": "archetype",
  "category": "Single Family House",
  "countries": ["Italy", "Greece"],
  "names": ["SFH_Italy_default", "SFH_Greece_default"]
}
```
Required fields:
- `mode`
- `category`
- `countries` (non-empty list)
- `names` (non-empty list)

#### B) `mode="custom"`
```json
{
  "mode": "custom",
  "buildings": [
    {"name": "B1", "bui": { ... }, "system": { ... }},
    {"name": "B2", "bui": { ... }, "system": { ... }}
  ]
}
```
Required fields:
- `mode`
- `buildings` (non-empty list)
- for each building: `name`, `bui`, `system`

### Response (200)
```json
{
  "status": "completed",
  "mode": "archetype" | "custom",
  "n_buildings": 2,
  "results": [
    {
      "name": "...",
      "status": "ok" | "error",
      "summary": { ... },
      "hourly": { ... },
      "annual": { ... },
      "system": { ... }
    }
  ],
  "summary": [
    {"name": "...", "heating_kWh": ..., "cooling_kWh": ..., "area": ...},
    ...
  ]
}
```

---

## 6) POST `/report`
**Purpose**: generate an HTML report (statistical analysis and charts) from an hourly time series.

### Body JSON (required)
```json
{
  "hourly_sim": [ {"timestamp": "...", "Q_HC": ..., ...}, ... ],
  "building_area": 120.0
}
```
Required fields:
- `hourly_sim` (list of records; converted to `pandas.DataFrame`)
- `building_area` (float)

### Response (200)
- `text/html` (HTMLResponse)

---

## 7) POST `/bui/epc_update_u_values`
**Purpose**: update envelope U-values according to an EPC energy class (Greece).

### Query params
- `energy_class` (string, **required**, regex `^[ABCD]$`)
- `archetype` (bool, default `true`)

If `archetype=true`, these are **required**:
- `category`
- `country`
- `name`

Optional:
- `new_name` (if not provided it is generated as `"{base_name}_class_{energy_class}"`)

### Custom mode (`archetype=false`)
**Body JSON (required)**:
```json
{
  "bui": { ... }
}
```
Required field: `bui`.

### Effect
- `archetype=true`: creates a **new** archetype and adds it to `BUILDING_ARCHETYPES` in memory.
- `archetype=false`: modifies and returns the BUI without persistence.

### Response (200)
```json
{
  "source": "archetype" | "custom",
  "base_name": "...",
  "base_category": "...",
  "base_country": "...",
  "energy_class": "A" | "B" | "C" | "D",
  "new_archetype_name": "..." | null,
  "bui": { ... }
}
```

---

## 8) POST `/ecm_application`
**Purpose**: generate all non-empty combinations of requested U-value interventions (roof/walls/windows) and simulate each scenario with ISO 52016.

### Query params
- `archetype` (bool, default `true`)
- `weather_source` (string, default `pvgis`, values: `pvgis` | `epw`)
- `u_wall` (float, optional)
- `u_roof` (float, optional)
- `u_window` (float, optional)

Required rule:
- you must specify **at least one** of `u_wall`, `u_roof`, `u_window`.

If `archetype=true`, these are **required**:
- `category`
- `country`
- `name`

### Custom mode (`archetype=false`)
**Content-Type**: `multipart/form-data`

Required field:
- `bui_json` (JSON string)

### EPW weather
If `weather_source=epw` then **required**:
- `epw_file` (file)

### Response (200)
```json
{
  "source": "archetype" | "custom",
  "name": "...",
  "category": "...",
  "country": "...",
  "weather_source": "pvgis" | "epw",
  "u_values_requested": {"roof": ..., "wall": ..., "window": ...},
  "n_scenarios": 3,
  "scenarios": [
    {
      "scenario_id": "roof+wall",
      "description": "roof U=... , walls U=...",
      "u_values": {"roof": ..., "wall": ..., "window": ...},
      "results": {
        "hourly_building": { ... },
        "annual_building": { ... }
      }
    }
  ]
}
```

---

## 9) GET `/emission-factors`
**Purpose**: returns emission factors (kgCO2eq/kWh) for a country.

### Query params
- `country` (string, optional, default `IT`)

Expected values in code:
- `IT`, `EU`, `DE`

### Response (200)
```json
{
  "country": "IT",
  "emission_factors_kg_co2eq_per_kwh": {
    "grid_electricity": 0.280,
    "natural_gas": 0.202,
    ...
  },
  "sources": ["grid_electricity", "natural_gas", ...]
}
```

---

## 10) POST `/calculate`
**Purpose**: compute CO2e for a single scenario.

### Body JSON (required)
Schema (Pydantic `ScenarioInput`):
```json
{
  "name": "Scenario name",
  "energy_source": "natural_gas",
  "annual_consumption_kwh": 20000,
  "country": "IT"
}
```
Required fields:
- `name`
- `energy_source`
- `annual_consumption_kwh` (> 0)

Optional field:
- `country` (default `IT`)

Allowed values for `energy_source` (enum `EnergySource`):
- `grid_electricity`
- `natural_gas`
- `lpg`
- `diesel`
- `biomass`
- `district_heating`
- `solar_pv`
- `wind`
- `heat_pump_electric`

### Response (200)
Schema (Pydantic `EmissionResult`):
```json
{
  "name": "Scenario name",
  "energy_source": "natural_gas",
  "annual_consumption_kwh": 20000,
  "emission_factor_kg_per_kwh": 0.202,
  "annual_emissions_kg_co2eq": 4040.0,
  "annual_emissions_ton_co2eq": 4.04,
  "equivalent_trees": 192,
  "equivalent_km_car": 33666
}
```

---

## 11) POST `/compare`
**Purpose**: compare N scenarios in terms of emissions. The first scenario in the list is the **baseline**.

### Body JSON (required)
Schema (Pydantic `MultiScenarioInput`):
```json
{
  "scenarios": [
    {
      "name": "Baseline",
      "energy_source": "natural_gas",
      "annual_consumption_kwh": 20000,
      "country": "IT"
    },
    {
      "name": "Heat pump",
      "energy_source": "heat_pump_electric",
      "annual_consumption_kwh": 5000,
      "country": "IT"
    }
  ]
}
```
Constraints:
- `scenarios` must contain **at least 2 items**

### Response (200)
Schema (Pydantic `ComparisonResult`):
```json
{
  "baseline": { ... EmissionResult ... },
  "scenarios": [ ... EmissionResult ... ],
  "best_scenario": "Heat pump",
  "savings": [
    {
      "scenario_name": "Heat pump",
      "absolute_kg_co2eq": 3200.0,
      "absolute_ton_co2eq": 3.2,
      "percentage": 79.2
    }
  ]
}
```

---

## 12) POST `/calculate-intervention`
**Purpose**: compute the CO2e impact of a retrofit intervention, with consumption reduction and/or source change.

### Body JSON (required)
Schema (Pydantic `InterventionInput`):
```json
{
  "current_consumption_kwh": 20000,
  "current_source": "natural_gas",
  "energy_reduction_percentage": 30,
  "new_source": "heat_pump_electric",
  "new_consumption_kwh": 5000,
  "country": "IT"
}
```
Required fields:
- `current_consumption_kwh` (> 0)
- `current_source`

Optional fields:
- `energy_reduction_percentage` (0..100, default 0)
- `new_source` (if missing, the current source is kept)
- `new_consumption_kwh` (if missing, it is calculated by applying `energy_reduction_percentage`)
- `country` (default `IT`)

### Response (200)
```json
{
  "intervention_summary": {
    "energy_reduction": "30%",
    "source_change": "natural_gas -> heat_pump_electric",
    "consumption_change": "20000 -> 5000 kWh/year"
  },
  "current_scenario": {
    "emissions_kg_co2eq": 4040.0,
    "emissions_ton_co2eq": 4.04
  },
  "future_scenario": {
    "emissions_kg_co2eq": 350.0,
    "emissions_ton_co2eq": 0.35
  },
  "savings": {
    "absolute_kg_co2eq": 3690.0,
    "absolute_ton_co2eq": 3.69,
    "percentage": 91.3
  },
  "environmental_impact": {
    "trees_saved": 175,
    "km_car_avoided": 30750
  }
}
```

---

# Health endpoints (router)

`main.py` includes `app.include_router(health.router)`. Based on your implementation in `relife_forecasting/routes/health.py`, you may have endpoints like:

- `GET /health`
- `GET /ready`
- `GET /live`

For a complete description:
- open Swagger `GET /docs` and check the paths under the (typically) `health` tag
- or document the contents of `health.py` directly.
