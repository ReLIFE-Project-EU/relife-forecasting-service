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
