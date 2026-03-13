# `ecm_application_api.py` Workflow

This document explains, in English, the full execution flow implemented in `ecm_application_api.py`.

## Purpose

The script is a sequential client/orchestrator for the ReLIFE forecasting service. It can:

1. Run a baseline simulation.
2. Run one or more renovation scenarios.
3. Call either the `/ecm_application` endpoint or the `/run/iso52016-uni11300-pv` endpoint, depending on the scenario type.
4. Save hourly and annual outputs as CSV files.
5. Build the HTML report locally from the results already collected.
6. Print a final summary table in the terminal.

## Main Configuration Blocks

The script is controlled by a set of constants near the top of the file:

- API endpoints: `BASE_URL`, `ECM_ENDPOINT`, `REPORT_ENDPOINT`, `UNI_PV_ENDPOINT`
- output folders: `RESULTS_DIR`, `BUILDING_EXAMPLES_DIR`
- building selection: `ARCHETYPE`, `CATEGORY`, `COUNTRY`, `ARCHETYPE_NAME`
- weather selection: `WEATHER_SOURCE`, `EPW_PATH`
- ECM parameters: `ECM_OPTIONS`, `U_WALL`, `U_ROOF`, `U_WINDOW`, `U_SLAB`
- generation parameters: `USE_HEAT_PUMP`, `HEAT_PUMP_COP_DEFAULT`, `HEAT_PUMP_COP_BY_COMBO`
- report switches: `INCLUDE_BASELINE`, `GENERATE_REPORT`, `REPORT_TITLE`
- execution mode: `USE_RENOVATION_SCENARIO_LIBRARY`
- optional single report target: `REPORT_SCENARIO_NAME`
- PV defaults: `DEFAULT_PV_CONFIG`

## Supported Execution Modes

The script has two main execution modes.

### 1. Scenario library mode

If `USE_RENOVATION_SCENARIO_LIBRARY = True`, the script reads the `RENOVATION_SCENARIOS` list and executes every scenario defined there.

Typical examples:

- envelope-only scenario, such as `deep_envelope`
- generation-only scenario, such as `condensing_boiler`
- PV scenario, such as `pv_only`
- mixed scenarios with envelope + heat pump + PV

### 2. Direct ECM option mode

If `USE_RENOVATION_SCENARIO_LIBRARY = False`, the script uses `ECM_OPTIONS` and related U-values to build:

- one baseline task
- one combined ECM task containing the selected envelope measures and optional heat pump

## Step-by-Step Execution Flow

### Step 1. Import helpers and define configuration

The script imports:

- `requests` for HTTP calls
- `pandas` and `numpy` for data handling
- HTML report builders from `ecm_report_html.py`
- local building archetypes from `building_examples`

Then it defines all runtime constants and the `RenovationScenario` dataclass.

### Step 2. Prepare utility functions

Several helper functions are defined before any API call happens:

- `slugify`: creates filesystem-safe names
- `ensure_dir`: creates output folders
- `json_safe`: converts NumPy values to JSON-safe Python values
- `save_bui_to_folder`: stores modified BUI JSON files locally
- `build_name_file`: creates CSV output paths
- `build_report_file`: creates HTML report output paths
- `to_dataframe`: converts payloads to pandas DataFrames
- `extract_envelope_elements`: keeps only `wall`, `roof`, `window`, `slab`
- `generate_ecm_combinations`: generates all subsets, though the current main flow is sequential and task-based

### Step 3. Classify surfaces and optionally modify the BUI

For custom buildings or local PV runs, the script can modify the building envelope locally.

This happens through:

- `classify_surface`: identifies whether a surface is a wall, roof, window, or slab
- `apply_u_values_to_BUI`: applies the configured U-values to matching surfaces and saves the modified BUI JSON for traceability

### Step 4. Build an `ApiTask`

Every simulation is represented by the `ApiTask` dataclass.

Each task contains:

- scenario identity
- active ECM combination
- weather source and optional EPW path
- output directory
- archetype or custom-building information
- generation flags such as heat pump or condensing boiler
- optional PV configuration

### Step 5. Choose the correct runner for each scenario

There are two execution runners.

#### Runner A: `call_ecm_application`

This runner is used for:

- baseline
- envelope scenarios
- heat-pump scenarios without PV
- condensing-boiler or generation-only scenarios handled by `/ecm_application`

Internal flow:

1. Build query parameters from the task.
2. Add `baseline_only=true` for the baseline task.
3. Add `scenario_elements=...` for envelope scenarios.
4. Add `use_heat_pump`, `heat_pump_cop`, `uni_generation_mode`, and `uni_eta_generation` when needed.
5. Add archetype identifiers or send `bui_json` for custom mode.
6. Attach the EPW file if `weather_source="epw"`.
7. Call `POST /ecm_application`.
8. Read the JSON response and select the final scenario returned by the endpoint.
9. Extract `hourly_building`, `annual_building`, and `primary_energy_uni11300`.
10. Save hourly and annual CSV files.
11. Build a `report_context` block containing:
    - building metadata
    - scenario metadata
    - hourly building results
    - UNI/TS 11300 results

#### Runner B: `call_iso52016_uni11300_pv`

This runner is used for scenarios containing PV.

Internal flow:

1. Resolve the local reference inputs through `_resolve_local_reference_inputs`.
2. Load either:
   - the selected archetype BUI and UNI configuration, or
   - the custom BUI JSON plus a default UNI configuration
3. Apply envelope U-values locally if the scenario contains envelope ECMs.
4. Build the UNI configuration using `_build_uni_config_for_task`.
5. Build the full request payload with:
   - `bui`
   - `pv`
   - `uni11300`
   - `return_hourly_building=true`
   - optional heat pump flags
6. Call `POST /run/iso52016-uni11300-pv`.
7. Extract:
   - building hourly results
   - UNI/TS 11300 results
   - PV/HP hourly and annual summaries
8. Build a compact annual DataFrame with `_build_integrated_annual_frame`.
9. Save:
   - building hourly CSV
   - annual CSV
   - optional PV hourly CSV
10. Build a `report_context` block so PV scenarios can also be included in the local HTML report.

### Step 6. Build the scenario combination tag

The function `build_scenario_combo` creates the logical scenario tag used for naming and selection.

It includes:

- envelope elements
- `heat_pump` if active
- `pv` if active
- `condensing_boiler` if requested
- `eta_generation` if a custom UNI generation efficiency is used

### Step 7. Build the report data locally

The script no longer depends on `/ecm_application/report` during the normal `main()` execution path.

Instead, `build_report_html_from_results`:

1. filters successful results
2. finds the baseline result
3. selects one scenario or multiple scenarios, depending on:
   - `scenario_combo`
   - `scenario_name`
   - `scenario_names`
4. normalizes scenario labels through `_normalize_report_scenario_context`
5. passes the collected contexts to the HTML builder

There are two report outcomes:

- single-scenario report through `build_ecm_comparison_report_html`
- multi-scenario report through `build_ecm_multi_scenario_report_html`

The current script defaults to a multi-scenario report in scenario-library mode when `REPORT_SCENARIO_NAME` is `None`.

### Step 8. Run all predefined renovation scenarios

If scenario-library mode is enabled, `run_predefined_renovation_scenarios`:

1. creates the output directories
2. optionally creates a baseline task
3. loops over each `RenovationScenario`
4. converts the scenario to a combo with `build_scenario_combo`
5. detects whether the scenario is:
   - envelope-based
   - generation-only
   - PV-based
6. builds an `ApiTask`
7. routes it to:
   - `call_ecm_application`, or
   - `call_iso52016_uni11300_pv`
8. stores every result in a list
9. returns summary statistics:
   - total
   - successful
   - failed
   - total time
   - results
   - scenario names

### Step 9. Run the direct ECM sequential flow

If scenario-library mode is disabled, `run_ecm_api_sequential`:

1. validates the selected `ECM_OPTIONS`
2. checks that all required U-values are available
3. creates output directories
4. extracts envelope options
5. determines whether the heat pump is active
6. builds:
   - one baseline task
   - one combined ECM task
7. runs all tasks sequentially through `call_ecm_application`
8. returns summary statistics and the selected `scenario_combo`

### Step 10. Build the terminal summary table

`build_summary_table` converts the raw result dictionaries into a cleaner DataFrame with:

- scenario name
- runner type
- combo tag
- status
- elapsed time
- output file paths
- error message if present

### Step 11. Execute `main()`

`main()` is the top-level entry point.

Its flow is:

1. choose scenario-library mode or direct ECM mode
2. run the simulations sequentially
3. decide whether the report should be generated
4. choose the report selection logic:
   - one named scenario, or
   - all selected renovation scenarios
5. build the HTML report locally from the already collected results
6. save the report to disk
7. print:
   - success/failure counts
   - total execution time
   - scenario library names, if used
   - summary table
   - report path
   - error list, if any
8. return a dictionary containing:
   - `stats`
   - `report_path`
   - `report_error`

## Files Written by the Script

Depending on the scenario type, the script writes:

- hourly building CSV
- annual CSV
- optional PV hourly CSV
- optional modified BUI JSON
- HTML report

File names are generated from:

- building name
- scenario name or combo tag
- weather tag (`pvgis` or EPW file stem)

## Important Design Notes

- Execution is sequential, not multiprocessing.
- The baseline is usually computed only once per run.
- The HTML report is generated locally from the collected results.
- Multi-scenario reporting is supported when more than one renovation scenario is selected.
- PV scenarios are supported in the report because the PV runner now stores the same `report_context` structure used by ECM-only runs.
- `call_ecm_report` still exists as a helper, but it is not the normal path used by `main()`.

## Practical Summary

In practical terms, `ecm_application_api.py` does the following:

1. reads the configuration
2. creates simulation tasks
3. runs baseline and selected renovation scenarios one after another
4. saves CSV outputs
5. stores enough context to build the report without rerunning simulations
6. generates a local HTML report
7. prints and returns a final execution summary
