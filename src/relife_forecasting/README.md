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
  "system": { ... },
  "uni11300_input_example": { ... }
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
    "hourly_building": [ {"timestamp": "...", "Q_HC": ..., ...}, ... ],
    "primary_energy_uni11300": {
      "input_unit": "Wh",
      "ideal_unit": "kWh",
      "n_hours": 8760,
      "summary": { ... }
    }
  }
}
```
> Note: in code, `annual_building` and `hourly_system` are commented out to reduce response size.

---

## 5) POST `/primary-energy/uni11300`
**Purpose**: compute delivered and primary energy from hourly ideal loads (UNI/TS 11300).

Input JSON supports:
- Direct hourly list: `hourly_sim` or `hourly_building`
- Full `/simulate` response: `results.hourly_building`
- If needed, wrap any of the above in `simulate_result`

`Q_H` and `Q_C` are mapped to ideal loads (`q_ideal_heat`, `q_ideal_cool`).
By default, inputs are assumed in **Wh** (as returned by `/simulate`). Use `input_unit="kWh"` if already in kWh.

### Body JSON (required)
```json
{
  "hourly_sim": [
    {"timestamp": "2024-01-01 00:00:00", "Q_H": 2103.85, "Q_C": 0.0},
    {"timestamp": "2024-01-01 01:00:00", "Q_H": 1800.12, "Q_C": 0.0}
  ],
  "input_unit": "Wh",
  "heating_params": { ... },
  "cooling_params": { ... }
}
```

### Response (200)
```json
{
  "input_unit": "Wh",
  "ideal_unit": "kWh",
  "n_hours": 8760,
  "hourly_results": [ ... ],
  "summary": {
    "EP_total_kWh": 12345.67
  }
}
```

### Example call
```bash
curl -X POST \
  "http://127.0.0.1:9091/primary-energy/uni11300" \
  -H "Content-Type: application/json" \
  -d '{
    "hourly_sim": [
      {"timestamp": "2024-01-01 00:00:00", "Q_H": 2103.85, "Q_C": 0.0},
      {"timestamp": "2024-01-01 01:00:00", "Q_H": 1800.12, "Q_C": 0.0}
    ],
    "input_unit": "Wh"
  }'
```

---

## 6) GET `/primary-energy/uni11300/input-example`
**Purpose**: return a ready-to-use input payload example (including default parameters).

### Response (200)
```json
{
  "hourly_sim": [
    {"timestamp": "2024-01-01 00:00:00", "Q_H": 2103.85, "Q_C": 0.0},
    {"timestamp": "2024-01-01 01:00:00", "Q_H": 1800.12, "Q_C": 0.0}
  ],
  "input_unit": "Wh",
  "heating_params": { ... },
  "cooling_params": { ... }
}
```

### Example call
```bash
curl -X GET "http://127.0.0.1:9091/primary-energy/uni11300/input-example"
```

---

## 7) POST `/simulate/batch`
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

## 8) POST `/report`
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

## 9) POST `/bui/epc_update_u_values`
**Purpose**: update envelope U-values according to an EPC energy class (Greece).

### Query params
- `energy_class` (string, **required**, regex `^[ABCD]$`)
- `archetype` (bool, default `true`)
- `u_slab` (float, optional) — override U-value for slab to ground
- `use_heat_pump` (bool, default `false`) — update system to heat pump
- `heat_pump_cop` (float, default `3.2`) — COP used when `use_heat_pump=true`

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
  "bui": { ... },
  "system": { ... }
}
```
Required field: `bui`. `system` is required only if `use_heat_pump=true` in custom mode.

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
  "bui": { ... },
  "system": { ... }
}
```

---

## 10) POST `/ecm_application`
**Purpose**: generate all non-empty combinations of requested U-value interventions (roof/walls/windows) and simulate each scenario with ISO 52016.

### Query params
- `archetype` (bool, default `true`)
- `weather_source` (string, default `pvgis`, values: `pvgis` | `epw`)
- `u_wall` (float, optional)
- `u_roof` (float, optional)
- `u_window` (float, optional)
- `u_slab` (float, optional)
- `use_heat_pump` (bool, default `false`) — update system to heat pump
- `heat_pump_cop` (float, default `3.2`) — COP used when `use_heat_pump=true`

Required rule:
- you must specify **at least one** of `u_wall`, `u_roof`, `u_window`, `u_slab`.

If `archetype=true`, these are **required**:
- `category`
- `country`
- `name`

### Custom mode (`archetype=false`)
**Content-Type**: `multipart/form-data`

Required field:
- `bui_json` (JSON string)
- `system_json` (JSON string, required only if `use_heat_pump=true`)

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

## 11) GET `/emission-factors`
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

## 12) POST `/calculate`
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

## 13) POST `/compare`
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

## 14) POST `/calculate-intervention`
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
# ECM Application 

# `POST /ecm_application` — ECM envelope U-values scenario simulation

This endpoint runs **envelope retrofit scenario simulations** by changing **U-values** (thermal transmittance) of the building envelope and computing energy needs with **ISO 52016** (via `pybuildingenergy`).

It can:
- simulate **all combinations** of the requested U-value interventions (multi-scenario mode), and/or
- simulate **a single scenario only** (single-scenario mode) to reduce compute time and payload.

---

## What the endpoint does

### 1) Selects the base building model (BUI)
You can run the simulation using:

**A) An internal archetype (default)**  
- `archetype=true`
- Requires `category`, `country`, `name`
- The endpoint looks up the building in `BUILDING_ARCHETYPES` and uses `match["bui"]` as the baseline BUI.

**B) A custom BUI**
- `archetype=false`
- Requires `bui_json` (multipart form field) as a JSON string
- The endpoint parses it and uses it as baseline BUI.

> This endpoint uses only the **building envelope** (BUI). It does **not** run HVAC/system simulation (ISO 15316).  
> If `use_heat_pump=true`, the system is updated and returned in `system_variant`, but it does not affect ISO 52016 results.

---

### 2) Builds retrofit scenarios (combinations of envelope upgrades)
You can pass any subset of these U-values:
- `u_wall`: new U-value for **opaque vertical surfaces** (walls)
- `u_roof`: new U-value for **opaque horizontal surfaces** (roof)
- `u_window`: new U-value for **transparent vertical surfaces** (windows)
- `u_slab`: new U-value for **slab to ground** (opaque ground surface)

The endpoint calls:

```python
scenarios_spec = build_uvalue_scenarios(u_roof=u_roof, u_wall=u_wall, u_window=u_window, u_slab=u_slab)
```

This helper is expected to return a list of scenario specs (non-empty combinations), each containing:
- `id` (scenario identifier)
- `label` (human-readable description)
- `use_roof`, `use_wall`, `use_window`, `use_slab` (booleans)

Example: if you provide `u_wall` and `u_window`, scenarios typically include:
- wall only
- window only
- wall + window

---

### 3) Supports **single-scenario mode** (NEW)
Single-scenario mode lets you simulate only one scenario (or baseline only).

#### A) Baseline only
- `baseline_only=true`

Behavior:
- ignores `scenario_id` and `scenario_elements`
- forces `include_baseline=true`
- returns only baseline results

#### B) Select scenario by scenario id
- `scenario_id=<id>`

Behavior:
- filters generated scenarios to the one whose `spec["id"] == scenario_id`
- if not found returns HTTP 404 with available IDs

#### C) Select scenario by elements
- `scenario_elements=wall,window`
- accepted separators: comma, `+`, semicolon, spaces (e.g. `roof+wall`, `roof wall`)

Behavior:
- parses requested elements into a set (must be subset of `{roof, wall, window, slab}`)
- compares it to each generated scenario spec’s elements
- keeps exactly one matching scenario or returns HTTP 404 with a helpful list of available scenarios

Valid element names are: `roof`, `wall`, `window`, `slab` (also accepts `ground` as an alias).

---

### 4) Weather source
Choose weather input via:
- `weather_source=pvgis` (default): weather retrieved via PVGIS inside the simulation library
- `weather_source=epw`: upload an EPW weather file in `epw_file`

When EPW is used, the file is written to a temporary `.epw` file and deleted at the end.

---

### 5) Runs ISO 52016 simulation
For each scenario (and optionally baseline), the endpoint runs:

```python
pybui.ISO52016.Temperature_and_Energy_needs_calculation(
    bui_variant,
    weather_source="pvgis" or "epw",
    path_weather_file=epw_path (if epw),
    sankey_graph=False,
)
```

Retrofit variants are created using:

```python
apply_u_values_to_bui(
    base_bui,
    use_roof=spec["use_roof"],
    use_wall=spec["use_wall"],
    use_window=spec["use_window"],
    use_slab=spec["use_slab"],
    u_roof=u_roof,
    u_wall=u_wall,
    u_window=u_window,
    u_slab=u_slab,
)
```

---

## Endpoint - ecm_application

**Route:** `POST /ecm_application`  
**Tags:** `Simulation`

---

## Parameters

### Building selection
| Name | Location | Type | Default | Required | Description |
|------|----------|------|---------|----------|-------------|
| `archetype` | query | bool | `true` | yes | Use internal archetype vs custom BUI |
| `category` | query | str | `None` | if `archetype=true` | Archetype category |
| `country` | query | str | `None` | if `archetype=true` | Archetype country |
| `name` | query | str | `None` | if `archetype=true` | Archetype name |
| `bui_json` | form | str | `None` | if `archetype=false` | JSON string containing BUI |

### Weather
| Name | Location | Type | Default | Required | Description |
|------|----------|------|---------|----------|-------------|
| `weather_source` | query | str | `pvgis` | yes | `pvgis` or `epw` |
| `epw_file` | file | UploadFile | `None` | if `weather_source=epw` | EPW file upload |

### U-values
| Name | Location | Type | Default | Required | Description |
|------|----------|------|---------|----------|-------------|
| `u_wall` | query | float | `None` | optional | New wall U-value |
| `u_roof` | query | float | `None` | optional | New roof U-value |
| `u_window` | query | float | `None` | optional | New window U-value |
| `u_slab` | query | float | `None` | optional | New slab-to-ground U-value |

### System update (optional)
| Name | Location | Type | Default | Description |
|------|----------|------|---------|-------------|
| `use_heat_pump` | query | bool | `false` | If `true`, update system to heat pump |
| `heat_pump_cop` | query | float | `3.2` | COP used when `use_heat_pump=true` |

### Single-scenario mode (NEW)
| Name | Location | Type | Default | Description |
|------|----------|------|---------|-------------|
| `scenario_elements` | query | str | `None` | Select exactly one scenario (e.g. `wall,window`) |
| `scenario_id` | query | str | `None` | Select exactly one scenario by ID |
| `baseline_only` | query | bool | `false` | If `true`, simulate baseline only |

### Baseline inclusion
| Name | Location | Type | Default | Description |
|------|----------|------|---------|-------------|
| `include_baseline` | query | bool | `false` | If `true`, include baseline results alongside scenarios |

---

## Response

Top-level structure:

```json
{
  "source": "archetype|custom",
  "name": "...",
  "category": "...",
  "country": "...",
  "weather_source": "pvgis|epw",
  "u_values_requested": { "roof": 0.8, "wall": 0.5, "window": 1.0, "slab": 0.6 },
  "system_variant": { ... },
  "single_scenario_mode": {
    "baseline_only": false,
    "scenario_id": null,
    "scenario_elements": null
  },
  "n_scenarios": 3,
  "scenarios": [
    {
      "scenario_id": "baseline",
      "description": "Baseline (no changes)",
      "elements": [],
      "u_values": { "roof": null, "wall": null, "window": null },
      "results": {
        "hourly_building": [...],
        "annual_building": [...]
      }
    },
    {
      "scenario_id": "wall+window",
      "description": "...",
      "elements": ["wall", "window"],
      "u_values": { "roof": null, "wall": 0.5, "window": 1.0 },
      "results": {
        "hourly_building": [...],
        "annual_building": [...]
      }
    }
  ]
}
```

Each scenario includes:
- `elements`: which envelope parts were modified
- `u_values`: which U-values were effectively applied
- `results.hourly_building`: hourly time series (JSON-safe)
- `results.annual_building`: annual aggregated results (JSON-safe)

---

## Error cases

### `400 Bad Request`
- `archetype=true` without `category/country/name`
- `archetype=false` without `bui_json`
- invalid JSON in `bui_json`
- invalid `weather_source` (must be `pvgis` or `epw`)
- `weather_source=epw` without `epw_file`
- nothing to simulate:
  - no scenarios produced and `include_baseline=false`
  - unless `baseline_only=true`

### `404 Not Found`
- archetype not found in `BUILDING_ARCHETYPES`
- `scenario_id` not found among generated scenarios
- `scenario_elements` does not match any generated scenario  
  (response includes an `available` list with ids, elements, and labels)

---

## Examples

### 1) Multi-scenario (all combinations) + baseline, PVGIS
```bash
curl -X POST \
  "http://127.0.0.1:9091/ecm_application?archetype=true&category=Single%20Family%20House&country=Greece&name=SFH_Greece_1946_1969&weather_source=pvgis&u_wall=0.5&u_window=1.0&include_baseline=true"
```

### 2) Single scenario by elements (wall + window), PVGIS
```bash
curl -X POST \
  "http://127.0.0.1:9091/ecm_application?archetype=true&category=Single%20Family%20House&country=Greece&name=SFH_Greece_1946_1969&weather_source=pvgis&u_wall=0.5&u_window=1.0&scenario_elements=wall,window"
```

### 3) Baseline only, EPW
```bash
curl -X POST \
  "http://127.0.0.1:9091/ecm_application?archetype=true&category=Single%20Family%20House&country=Greece&name=SFH_Greece_1946_1969&weather_source=epw&baseline_only=true" \
  -F "epw_file=@src/relife_forecasting/epw_weather/GRC_Athens.167160_IWEC.epw"
```

---

## Notes / performance

- Multi-scenario mode can be expensive because it runs *one ISO 52016 simulation per scenario*.
- Single-scenario mode is designed to support:
  - fast API calls from clients that loop over combinations sequentially
  - “one scenario per request” workflows
- The response payload can be large because it includes hourly time series;


---

# `POST /ecm_application/run_sequential_save` — Sequential ECM run + save results to disk

This endpoint is a **server-side runner** that executes multiple envelope retrofit scenarios **sequentially** (no multiprocessing) and **saves results to disk** as CSV files (and optionally BUI JSON variants).  

It is designed to avoid returning huge hourly payloads over HTTP: instead it returns **file paths**, per-scenario status, and a summary.

---

## What the endpoint does

### 1) Determines the base building (archetype only)
Currently this endpoint supports **archetype mode only**:

- `archetype=true` (default)
- requires `category`, `country`, `name`
- finds a matching entry in `BUILDING_ARCHETYPES`
- uses `match["bui"]` as the baseline BUI
- uses `match["name"]` (or `name`) as `building_name` for filenames

If `archetype=false` the endpoint returns **HTTP 400** (it can be extended later to accept a `bui_json=Form(...)` input).

---

### 2) Determines which ECM elements to combine (`opts`)
You can explicitly set `ecm_options` or let the endpoint infer it from provided U-values:

**A) Explicit**
- `ecm_options="wall,window,roof,slab"` (comma-separated)

**B) Inferred**
- if `ecm_options` is omitted:
  - includes `"wall"` if `u_wall` is provided
  - includes `"roof"` if `u_roof` is provided
  - includes `"window"` if `u_window` is provided

Validation rules:
- allowed options: `wall`, `roof`, `window`
- if an option is present, the corresponding `u_*` must be provided
- if no options and `include_baseline=false` → HTTP 400 (“Nothing to run”)

---

### 3) Generates the scenario combinations
The endpoint builds the list of combinations:

```python
combos = _generate_combinations(opts, include_baseline=include_baseline)
```

This returns:
- baseline `[]` if `include_baseline=true`
- all non-empty subsets of `opts`

Example: `opts=["wall","window"]` and `include_baseline=true` →  
`[]`, `["wall"]`, `["window"]`, `["wall","window"]`

---

### 4) Handles weather source (PVGIS or EPW)
- `weather_source="pvgis"`: uses PVGIS weather via the simulation library.
- `weather_source="epw"`: requires file upload `epw_file`, stored once into a temporary `.epw` file and reused for all scenarios. The temp file is deleted at the end.

---

### 5) For each scenario (sequential loop)
For each combination:

1. **Builds a BUI variant**
   - baseline: `bui_variant = base_bui`
   - scenario: calls `apply_u_values_to_bui(...)` with flags:
     - `use_roof=("roof" in combo)`
     - `use_wall=("wall" in combo)`
     - `use_window=("window" in combo)`
     - `use_slab=("slab" in combo)`
     - and U-values `u_roof/u_wall/u_window/u_slab`

2. **Optionally saves BUI JSON variant**
   - if `save_bui=true`:
     - `_save_bui_variant(bui_variant, combo, bui_dir)`
     - filename pattern:
       - `BUI_<building>__baseline.json` (baseline)
       - `BUI_<building>__wall_window.json` (combo)

3. **Runs ISO 52016 simulation**
   - PVGIS:
     ```python
     pybui.ISO52016.Temperature_and_Energy_needs_calculation(
         bui_variant, weather_source="pvgis", sankey_graph=False
     )
     ```
   - EPW:
     ```python
     pybui.ISO52016.Temperature_and_Energy_needs_calculation(
         bui_variant, weather_source="epw", path_weather_file=epw_path, sankey_graph=False
     )
     ```

4. **Saves results to CSV**
   - hourly file name via `_build_name_file(...)`:
     - `<building>__<combo_tag>__<weather_tag>.csv`
   - annual file:
     - `<hourly_stem>__annual.csv`

5. **Records outcome**
   - `success`: includes file paths + elapsed time
   - `error`: includes exception + traceback + elapsed time (loop continues)

---

## Endpoint - run_sequential_save

**Route:** `POST /ecm_application/run_sequential_save`  
**Tags:** `Simulation`

This is intended for internal / batch workflows where the server has access to its own filesystem.

---

## Request parameters

### Building selection (archetype)
| Name | Location | Type | Default | Required | Description |
|------|----------|------|---------|----------|-------------|
| `archetype` | query | bool | `true` | yes | Must be `true` in current implementation |
| `category` | query | str | `None` | yes | Building category (archetype lookup) |
| `country` | query | str | `None` | yes | Country (archetype lookup) |
| `name` | query | str | `None` | yes | Archetype name (archetype lookup) |

### Weather
| Name | Location | Type | Default | Required | Description |
|------|----------|------|---------|----------|-------------|
| `weather_source` | query | str | `pvgis` | yes | `pvgis` or `epw` |
| `epw_file` | file | UploadFile | `None` | if `weather_source=epw` | EPW weather file |

### ECM controls
| Name | Location | Type | Default | Required | Description |
|------|----------|------|---------|----------|-------------|
| `ecm_options` | query | str | `None` | optional | Comma-separated list of elements to combine (wall, window, roof, slab). If omitted, inferred from provided U-values. |
| `u_wall` | query | float | `None` | required if `"wall"` in options | New wall U-value |
| `u_roof` | query | float | `None` | required if `"roof"` in options | New roof U-value |
| `u_window` | query | float | `None` | required if `"window"` in options | New window U-value |
| `u_slab` | query | float | `None` | required if `"slab"` in options | New slab-to-ground U-value |
| `include_baseline` | query | bool | `true` | optional | If true, baseline scenario is included |

### Saving controls
| Name | Location | Type | Default | Description |
|------|----------|------|---------|-------------|
| `output_dir` | query | str | `results/ecm_api` | Folder where CSV files are written |
| `save_bui` | query | bool | `true` | If true, save BUI JSON variants |
| `bui_dir` | query | str | `building_examples_ecm_api` | Folder where BUI JSON variants are written |

---

## Response schema

Top-level response:

```json
{
  "status": "completed",
  "source": "archetype",
  "building": { "category": "...", "country": "...", "name": "..." },
  "weather_source": "pvgis|epw",
  "u_values_requested": { "roof": 0.8, "wall": 0.5, "window": 1.0 },
  "ecm_options": ["wall", "window"],
  "include_baseline": true,
  "output_dir": "results/ecm_api",
  "bui_dir": "building_examples_ecm_api",
  "summary": {
    "total": 4,
    "successful": 4,
    "failed": 0,
    "total_time_s": 12.345
  },
  "results": [
    {
      "status": "success",
      "combo": ["wall", "window"],
      "combo_tag": "wall,window",
      "files": {
        "hourly_csv": "...",
        "annual_csv": "...",
        "bui_json": "..."
      },
      "elapsed_s": 3.21
    }
  ]
}
```

### Per-scenario result item
- `status`: `"success"` or `"error"`
- `combo`: list of elements applied (empty list means baseline)
- `combo_tag`: human-readable tag (`"BASELINE"` or `"wall,window"`)
- `files` (success):
  - `hourly_csv`
  - `annual_csv`
  - `bui_json` (may be `null` if `save_bui=false`)
- `error` and `traceback` (error only)
- `elapsed_s`: time spent for that scenario

---

## File naming conventions

### Hourly CSV
Built by `_build_name_file(...)`:

```
<building_slug>__<combo_tag>__<weather_tag>.csv
```

- `combo_tag` is `"baseline"` or joined sorted elements (e.g. `wall_window`)
- `weather_tag` is:
  - `"pvgis"` if `weather_source="pvgis"`
  - EPW filename stem if `weather_source="epw"` and EPW exists

### Annual CSV
```
<hourly_stem>__annual.csv
```

### BUI JSON (if enabled)
Written by `_save_bui_variant(...)`:

```
BUI_<building_slug>__<combo_tag>.json
```

---

## Error behavior
- The endpoint is **best-effort**: one scenario failing does **not** stop the others.
- EPW temp file cleanup happens in `finally`.
- Typical errors:
  - missing archetype inputs
  - invalid `ecm_options`
  - `weather_source=epw` without `epw_file`
  - simulation exceptions inside `pybuildingenergy`

---

## Example calls

### PVGIS mode
```bash
curl -X POST \
  "http://127.0.0.1:9091/ecm_application/run_sequential_save?archetype=true&category=Single%20Family%20House&country=Greece&name=SFH_Greece_1946_1969&weather_source=pvgis&ecm_options=wall,window&u_wall=0.5&u_window=1.0&include_baseline=true&output_dir=results/ecm_api&save_bui=true&bui_dir=building_examples_ecm"
```

### EPW mode (multipart upload)
```bash
curl -X POST \
  "http://127.0.0.1:9091/ecm_application/run_sequential_save?archetype=true&category=Single%20Family%20House&country=Greece&name=SFH_Greece_1946_1969&weather_source=epw&ecm_options=wall,window&u_wall=0.5&u_window=1.0&include_baseline=true" \
  -F "epw_file=@src/relife_forecasting/epw_weather/GRC_Athens.167160_IWEC.epw"
```

---

## Operational notes
- The returned file paths are on the **server filesystem**. If your tests run on a different machine/container, file existence assertions should be adapted.
- Because it writes to disk, ensure the process has permissions to create `output_dir` / `bui_dir`.
- This endpoint intentionally avoids returning hourly JSON to keep responses small and fast.
---
# Health endpoints (router)

`main.py` includes `app.include_router(health.router)`. Based on your implementation in `relife_forecasting/routes/health.py`, you may have endpoints like:

- `GET /health`
- `GET /ready`
- `GET /live`

For a complete description:
- open Swagger `GET /docs` and check the paths under the (typically) `health` tag
- or document the contents of `health.py` directly.
