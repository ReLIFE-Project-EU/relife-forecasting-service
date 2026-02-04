Main differences between the API clients:

- `ecm_application_api.py` (and `ecm_application_api_ecm_application.py`) call `POST /ecm_application`
  in "single-scenario" mode.
  It builds all ECM combinations locally (baseline + subsets) and performs one HTTP call per scenario.
  CSV outputs (hourly/annual) are saved by the client in the local folder (e.g. `results/ecm_api`).
  Saving modified BUIs is handled client-side (useful only in custom mode or after fetching `/building`).
  At the end `ecm_application_api.py` prints a **summary table** for each simulation (combo, status,
  scenario_id, elapsed time, size, heat pump COP, output files, and error if any).

- `ecm_application_api_run_sequential_save.py` calls `POST /ecm_application/run_sequential_save` once.
  ECM combinations are generated and executed sequentially on the server.
  Output files (CSV and BUI) are saved on the server filesystem, and the client receives only paths/summary.
  The client just prints the summary and a few result entries.

Example summary table output:
```
SUMMARY TABLE (per simulation)
 combo     status scenario_id  elapsed_s  size_kb  heat_pump_cop                                   file_hourly                                    file_annual                error
baseline  success    baseline      2.143   512.31           NaN  results/ecm_api/sfh...baseline.csv  results/ecm_api/sfh...baseline__annual.csv                   NaN
wall      success     wall          2.221   508.90           3.2  results/ecm_api/sfh...wall.csv      results/ecm_api/sfh...wall__annual.csv                       NaN
```
