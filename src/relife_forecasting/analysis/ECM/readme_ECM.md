Main differences between the API clients:

- `ecm_application_api_ecm_application.py` calls `POST /ecm_application` in "single-scenario" mode.
  It builds all ECM combinations locally (baseline + subsets) and performs one HTTP call per scenario.
  CSV outputs (hourly/annual) are saved by the client in the local folder (e.g. `results/ecm_api`).
  Saving modified BUIs is handled client-side (useful only in custom mode or after fetching `/building`).

- `ecm_application_api_run_sequential_save.py` calls `POST /ecm_application/run_sequential_save` once.
  ECM combinations are generated and executed sequentially on the server.
  Output files (CSV and BUI) are saved on the server filesystem, and the client receives only paths/summary.
  The client just prints the summary and a few result entries.
