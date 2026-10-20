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

Country archetypes (like `AT_SFH_1946-1969`) are generated in `building_examples.py` using data from Excel workbooks in `EU_new archetypes/`. The script `python scripts/check_archetypes_vs_excel.py` checks that the code matches these workbooks. When data is missing, the following assumptions are made:

- **Net floor area**: footprint × number of floors. Height = floors × 2.7 m. Volume = footprint × floors × 2.7 m.
- **Ventilation**: Each archetype has a constant air change rate per hour (ACH) by construction period: 0.90 (oldest) down to 0.55 (most recent), applied to volume.
- **Thermal bridges**: Total = delta-U × envelope area, with delta-U decreasing from 0.10 to 0.03 W/m²K over time.
- **Internal gains**: Based on EN 16798-1 standards. Occupants: 2.8 W/m² (houses), 4.2 (apartments); appliances: 2.4/3.0; lighting: 3.0. Heating: 5:00–23:00 (17°C setback); cooling follows occupancy; infiltration runs continuously.
- **System capacities**: Heating = 1.25 × design heat loss × (20°C – outdoor design temp, which is estimated from national heating degree days). Cooling: 100 W/m².
- **Solar absorptance**: 0.6 for both roof and walls, all directions.

## Archetype floor area (`floor_area`)

The archetype endpoints `POST /building`, `POST /simulate`, `POST /ecm_application` and `POST /ecm_application/report` accept an optional `floor_area` query parameter (m², > 0).

- Omitted, or equal to the archetype's `building.net_floor_area`: the archetype is used unchanged.
- Different from the archetype value: the BUI is rescaled with `scale_bui_floor_area()` (`routes/forecasting_service_functions.py`) before use. The archetype stored in `BUILDING_ARCHETYPES` is never modified.
- Ignored when `archetype=false` (custom BUIs are used as sent).

The building keeps the same footprint shape, `height` and `n_floors`. With `k = floor_area / archetype_floor_area` and `s = sqrt(k)`:

| BUI field                              | Change                             |
| -------------------------------------- | ---------------------------------- |
| `building.net_floor_area`              | set to `floor_area`                |
| roof and slab-to-ground `area`         | x k                                |
| `building.exposed_perimeter`           | x s                                |
| opaque vertical walls `area`           | x s (perimeter x unchanged height) |
| windows `area` and `width`             | x s (same window-to-wall ratio)    |
| `system_capacities` (heating, cooling) | x k                                |
| volume (floor area x height)           | x k, follows from the above        |

Walls and windows grow more slowly than the floor area, so the envelope per m² decreases and energy needs per m² drop for larger buildings. U-values, orientations, setpoints and `height` are not changed. Example:

```
POST /simulate?archetype=true&category=Single Family House&country=Austria&name=AT_SFH_1946-1969&floor_area=187.5&weather_source=pvgis
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