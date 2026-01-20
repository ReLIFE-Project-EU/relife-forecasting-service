"""
Example: parallel simulation of multiple building archetypes with pybuildingenergy.

- For each archetype in BUILDING_ARCHETYPES:
    * Run ISO 52016 (Temperature_and_Energy_needs_calculation)
    * Run the heating system simulation (HeatingSystemCalculator)
    * Return the DataFrames and a compact summary

Outputs:
    - results_by_archetype: dict
        {
            "archetype_name": {
                "hourly_building": df_hourly,
                "annual_building": df_annual,
                "hourly_system": df_system,
            },
            ...
        }

    - summary_df: summary DataFrame with one row per archetype
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import pybuildingenergy as pybui
from building_examples import BUILDING_ARCHETYPES


# ---------------------------
# Single-archetype simulation
# ---------------------------

def simulate_one_archetype(archetype: Dict[str, Any]) -> Tuple[str, Dict[str, pd.DataFrame], pd.Series]:
    """
    Simulate a single archetype.

    Returns:
        - archetype name
        - dict of DataFrames:
            {
                "hourly_building": df_hourly,
                "annual_building": df_annual,
                "hourly_system": df_system,
            }
        - a Series with the main KPIs (used to build summary_df)
    """
    # Extract the core inputs for this archetype
    name = archetype["name"]
    bui = archetype["bui"]
    system = archetype["system"]

    # --- 1) ISO 52016 simulation (building) ---
    # This returns an hourly time series and an annual results DataFrame.
    hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
        bui,
        weather_source="pvgis",  # Change if you want EPW or other sources supported by your setup.
    )

    # --- 2) System simulation (ISO 15316) ---
    # The system calculator consumes the hourly building simulation outputs.
    calc = pybui.HeatingSystemCalculator(system)
    calc.load_csv_data(hourly_sim)
    df_system = calc.run_timeseries()

    # --- 3) Compact KPIs for comparison across archetypes ---
    # NOTE: Adjust column names (e.g., "Q_HC") to match your pybuildingenergy version.
    # Convention assumed here:
    #   - Positive Q_HC = heating demand (Wh)
    #   - Negative Q_HC = cooling demand (Wh)
    heating_kWh = hourly_sim.loc[hourly_sim["Q_HC"] > 0, "Q_HC"].sum() / 1000.0
    cooling_kWh = -hourly_sim.loc[hourly_sim["Q_HC"] < 0, "Q_HC"].sum() / 1000.0

    # Use net_floor_area as treated area (replace with treated_floor_area if your input model provides it).
    treated_floor_area = bui["building"].get("net_floor_area", np.nan)

    # Compute specific energy (kWh/m2). Guard against missing/zero areas.
    heating_kWh_m2 = heating_kWh / treated_floor_area if treated_floor_area else np.nan
    cooling_kWh_m2 = cooling_kWh / treated_floor_area if treated_floor_area else np.nan

    # Build a compact summary Series (one row later in summary_df).
    summary = pd.Series(
        {
            "name": name,
            "category": archetype.get("category"),
            "country": archetype.get("country"),
            "heating_kWh": heating_kWh,
            "cooling_kWh": cooling_kWh,
            "treated_floor_area": treated_floor_area,
            "heating_kWh_m2": heating_kWh_m2,
            "cooling_kWh_m2": cooling_kWh_m2,
        }
    )

    # Pack all DataFrames for this archetype.
    dfs = {
        "hourly_building": hourly_sim,
        "annual_building": annual_results_df,
        "hourly_system": df_system,
    }

    return name, dfs, summary


# -----------------------------------
# Parallel batch simulation function
# -----------------------------------

def simulate_archetypes_in_parallel(
    archetypes: list[Dict[str, Any]],
    max_workers: int | None = None,
):
    """
    Run multiple archetype simulations in parallel.

    Args:
        archetypes: list of archetypes (BUILDING_ARCHETYPES or a subset)
        max_workers: number of worker processes. If None, uses ProcessPoolExecutor default.

    Returns:
        results_by_archetype: dict[name] -> dict of DataFrames
        summary_df: DataFrame with one row per archetype (compact KPIs)
    """
    results_by_archetype: Dict[str, Dict[str, pd.DataFrame]] = {}
    summaries: list[pd.Series] = []

    # ProcessPoolExecutor runs each archetype in a separate process (useful for CPU-bound tasks).
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit one job per archetype; keep a mapping from Future -> archetype name for error reporting.
        futures = {executor.submit(simulate_one_archetype, arch): arch["name"] for arch in archetypes}

        # Collect results as they complete (order is not guaranteed).
        for future in as_completed(futures):
            arch_name = futures[future]
            try:
                name, dfs, summary = future.result()
                results_by_archetype[name] = dfs
                summaries.append(summary)
                print(f"[OK] Simulation completed for archetype: {name}")
            except Exception as e:
                # Continue running other archetypes even if one fails.
                print(f"[ERR] Simulation failed for archetype {arch_name}: {e}")

    # Build a summary DataFrame with one row per archetype.
    if summaries:
        summary_df = pd.DataFrame(summaries).set_index("name")
    else:
        # Return an empty-but-structured DataFrame if all simulations failed.
        summary_df = pd.DataFrame(
            columns=[
                "name",
                "category",
                "country",
                "heating_kWh",
                "cooling_kWh",
                "treated_floor_area",
                "heating_kWh_m2",
                "cooling_kWh_m2",
            ]
        ).set_index("name", drop=False)

    return results_by_archetype, summary_df


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Example: simulate all archetypes defined in BUILDING_ARCHETYPES.
    # If you want only a subset, filter the list, e.g.:
    # archetypes_to_run = [a for a in BUILDING_ARCHETYPES if a.get("country") == "Italy"]
    archetypes_to_run = BUILDING_ARCHETYPES

    results, summary_df = simulate_archetypes_in_parallel(
        archetypes_to_run,
        max_workers=None,  # Or set a fixed number, e.g. 4
    )

    # Print compact summary
    print("\n===== SUMMARY =====")
    print(summary_df)

    # Example: access DataFrames for one archetype
    example_name = next(iter(results)) if results else None
    if example_name:
        df_hourly = results[example_name]["hourly_building"]
        df_annual = results[example_name]["annual_building"]
        df_system = results[example_name]["hourly_system"]

        print(f"\nExample results for: {example_name}")
        print("hourly_building:", df_hourly.shape)
        print("annual_building:", df_annual.shape)
        print("hourly_system:", df_system.shape)

        # Optionally save to CSV/parquet:
        # df_hourly.to_csv(f"hourly_{example_name}.csv", index=False)
        # df_annual.to_csv(f"annual_{example_name}.csv", index=False)
        # df_system.to_csv(f"system_{example_name}.csv", index=False)
