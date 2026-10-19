# ReLIFE Forecasting Service

This service estimates how renovation changes a building's heating, cooling, electricity use, emissions, and health impacts from indoor temperatures.

## Understanding the simulation

Start with a building description or an archetype: a predefined example representing a building type and construction period. The baseline is the building before renovation; scenarios apply changes such as insulation, new windows, or a heat pump.

The building envelope means its walls, roof, floor, and windows. Their [U-values](https://greenheattoolkit.energysavingtrust.org.uk/t/insulation-toolkit/why-do-homes-and-commercial-buildings-need-insulation/heat-loss/) describe how readily heat passes through them; lower values mean better insulation. ECM means energy conservation measure, such as insulating a wall.

Weather comes from [PVGIS](https://joint-research-centre.ec.europa.eu/pvgis-online-tool_en), the European Commission's solar and weather data service, or an uploaded EPW (EnergyPlus Weather) file.

The calculations distinguish three energy quantities:

| Quantity              | Meaning                                                                                                                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Heating/cooling needs | Heat added or removed to maintain the chosen indoor temperatures. `pybuildingenergy` uses the [ISO 52016](https://www.iso.org/standard/65696.html) standard to calculate these hour by hour. |
| Delivered energy      | Fuel or electricity the heating/cooling equipment needs. UNI/TS 11300 methods account for equipment efficiency and losses.                                                                   |
| Primary energy        | Delivered energy multiplied by factors representing energy used to supply that fuel or electricity.                                                                                          |

## Using the results

- `POST /simulate`: simulate a building and its systems.
- `POST /ecm_application`: compare renovation scenarios.
- `POST /run/iso52016-uni11300-pv`: combine building, system, and solar-panel calculations. PV means photovoltaic electricity generation.
- `POST /ecm_application/daly`: estimate heat/cold health benefits of envelope renovations. Avoided disability-adjusted life years (DALYs) measure estimated healthy life saved.

Results include hourly values, annual totals, and scenario comparisons, all dependent on building, weather, equipment, and health assumptions. See the [scenario workflow](src/relife_forecasting/analysis/ECM/readme_ecm_application_api.md) for examples and reports.

## Run locally

Requires Python 3.11 and `uv`:

```bash
uv sync --frozen
uv run --frozen run-service
```

Open [API documentation](http://localhost:9090/docs); `GET /health` checks availability. PVGIS requires internet access.

`API_HOST`, `API_PORT`, and `API_WORKERS` default to `0.0.0.0`, `9090`, and `1`; see [configuration](src/relife_forecasting/__init__.py). Supabase/Keycloak credentials are not required.

Run tests with `uv run --frozen pytest`. Licensed under [EUPL-1.2](LICENSE).

## Archetype assumptions

The country archetypes (`<CC>_SFH|MFH|AB_<period>`, e.g. `AT_SFH_1946-1969`) are built in
`building_examples.py` from the workbooks in `EU_new archetypes/` (geometry, U-values, g-value,
generation efficiencies). `python scripts/check_archetypes_vs_excel.py` verifies code against workbooks.
Values the workbooks do not provide are assumptions:

- **Net floor area** is per building: footprint x floors. `building.height` = floors x 2.7 m and
  `building.volume` = footprint x floors x 2.7 m.
- **Ventilation**: pybuildingenergy >= 2.1 only applies ventilation if `building_parameters.ventilation`
  is present, so each archetype declares a `constant_ach` component (rate by period, applied to
  `building.volume`): 0.90 (0-1945), 0.80, 0.75, 0.70, 0.65, 0.60, 0.55 (2011-now) 1/h.
- **Thermal bridges**: total `thermal_bridge_heat_W_K` = delta-U x gross envelope area, with delta-U
  0.10 (to 1969), 0.08 (1970-89), 0.06 (1990s), 0.05 (2000-10), 0.03 (2011-now) W/m2K.
- **Internal gains and profiles**: EN 16798-1 residential values and schedules. Occupants
  2.8 W/m2 (houses) / 4.2 (apartments), appliances 2.4 / 3.0, lighting 3.0. Heating 05:00-23:00 with
  17 C setback; cooling follows occupancy; infiltration is continuous.
- **Capacities**: heating = 1.25 x design heat loss x (20 C - design outdoor temperature), the latter
  estimated from the country's heating degree days (Eurostat 2014-2023); cooling = 100 W/m2.
- **Solar absorptance** of roof and walls is 0.6 on every orientation.

The generic archetypes (`SFH_0_1945`, `SFH_1946_1969`, `SFH_1970_1979`, `SFH_Greece_*`, `SFH_Italy_*`)
use the same period values; their geometry is a simplified placeholder.

## Archetype floor area (`floor_area`)

The archetype endpoints `POST /building`, `POST /simulate`, `POST /ecm_application` and
`POST /ecm_application/report` accept an optional `floor_area` query parameter (m², > 0).

- Omitted, or equal to the archetype's `building.net_floor_area`: the archetype is used unchanged.
- Different from the archetype value: the BUI is rescaled with
  `scale_bui_floor_area()` (`routes/forecasting_service_functions.py`) before use.
  The archetype stored in `BUILDING_ARCHETYPES` is never modified.
- Ignored when `archetype=false` (custom BUIs are used as sent).

The building keeps the same footprint shape, `height` and `n_floors`. With
`k = floor_area / archetype_floor_area` and `s = sqrt(k)`:

| BUI field | Change |
|---|---|
| `building.net_floor_area` | set to `floor_area` |
| roof and slab-to-ground `area` | x k |
| `building.exposed_perimeter` | x s |
| opaque vertical walls `area` | x s (perimeter x unchanged height) |
| windows `area` and `width` | x s (same window-to-wall ratio) |
| `system_capacities` (heating, cooling) | x k |
| volume (floor area x height) | x k, follows from the above |

Walls and windows grow more slowly than the floor area, so the envelope per m² decreases and
energy needs per m² drop for larger buildings. U-values, orientations, setpoints and
`height` are not changed. Example:

```
POST /simulate?archetype=true&category=Single Family House&country=Austria&name=SFH_1946_1969&floor_area=187.5&weather_source=pvgis
```

Responses of `POST /building` return the rescaled BUI, so it can be inspected before simulating.

Ready-made comparison (archetype as stored vs. modified floor area) against a running API:

```
uv run uvicorn relife_forecasting.main:app --port 8000
python scripts/run_floor_area_comparison.py                       # PVGIS weather, floor_area=187.5
python scripts/run_floor_area_comparison.py --floor-area 160 --weather epw \
    --epw src/relife_forecasting/epw_weather/GRC_Athens.167160_IWEC.epw
```

See [scripts/run_floor_area_comparison.py](scripts/run_floor_area_comparison.py).

## Simulation library and Sankey

The service uses `pybuildingenergy==2.1.0`. The ISO 52016 Sankey diagram is never generated:
`run_iso52016_simulation` always calls the library with `sankey_graph=False`.
