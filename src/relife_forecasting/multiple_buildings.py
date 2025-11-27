"""
Esempio di simulazione in parallelo di più archetipi con pybuildingenergy.

- Per ogni archetipo in BUILDING_ARCHETYPES:
    * Simula ISO 52016 (Temperature_and_Energy_needs_calculation)
    * Simula il sistema di riscaldamento (HeatingSystemCalculator)
    * Restituisce i DataFrame e un riepilogo sintetico

Risultati:
    - results_by_archetype: dict
        {
            "nome_archetipo": {
                "hourly_building": df_hourly,
                "annual_building": df_annual,
                "hourly_system": df_system,
            },
            ...
        }

    - summary_df: DataFrame di sintesi con una riga per archetipo
"""

from __future__ import annotations

import os
from typing import Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import pybuildingenergy as pybui
from building_examples import BUILDING_ARCHETYPES


# ---------------------------
# Funzione di simulazione
# ---------------------------

def simulate_one_archetype(archetype: Dict[str, Any]) -> Tuple[str, Dict[str, pd.DataFrame], pd.Series]:
    """
    Simula un singolo archetipo.

    Restituisce:
        - nome_archetipo
        - dict con i DataFrame:
            {
                "hourly_building": df_hourly,
                "annual_building": df_annual,
                "hourly_system": df_system,
            }
        - una Series con i principali indicatori (usata poi per costruire summary_df)
    """
    name = archetype["name"]
    bui = archetype["bui"]
    system = archetype["system"]

    # --- 1) Simulazione ISO 52016 (building) ---
    hourly_sim, annual_results_df = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
        bui,
        weather_source="pvgis",  # adatta se vuoi EPW ecc.
    )

    # --- 2) Simulazione sistema (ISO 15316) ---
    calc = pybui.HeatingSystemCalculator(system)
    calc.load_csv_data(hourly_sim)
    df_system = calc.run_timeseries()

    # --- 3) Indicatori sintetici per confronto ---
    # NB: adatta i nomi delle colonne Q_HC / ecc. alla tua versione
    heating_kWh = hourly_sim.loc[hourly_sim["Q_HC"] > 0, "Q_HC"].sum() / 1000.0
    cooling_kWh = -hourly_sim.loc[hourly_sim["Q_HC"] < 0, "Q_HC"].sum() / 1000.0

    # uso net_floor_area come superficie trattata (adatta se hai treated_floor_area)
    treated_floor_area = bui["building"].get("net_floor_area", np.nan)

    heating_kWh_m2 = heating_kWh / treated_floor_area if treated_floor_area else np.nan
    cooling_kWh_m2 = cooling_kWh / treated_floor_area if treated_floor_area else np.nan

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

    dfs = {
        "hourly_building": hourly_sim,
        "annual_building": annual_results_df,
        "hourly_system": df_system,
    }

    return name, dfs, summary


# ---------------------------
# Funzione batch in parallelo
# ---------------------------

def simulate_archetypes_in_parallel(
    archetypes: list[Dict[str, Any]],
    max_workers: int | None = None,
):
    """
    Lancia la simulazione di più archetipi in parallelo.

    Args:
        archetypes: lista di archetipi (BUILDING_ARCHETYPES o un sottoinsieme)
        max_workers: numero di processi. Se None, usa il default di ProcessPoolExecutor.

    Returns:
        results_by_archetype: dict[name] -> dict di DataFrame
        summary_df: DataFrame con una riga per archetipo (indicatori sintetici)
    """
    results_by_archetype: Dict[str, Dict[str, pd.DataFrame]] = {}
    summaries = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(simulate_one_archetype, arch): arch["name"]
            for arch in archetypes
        }

        for future in as_completed(futures):
            arch_name = futures[future]
            try:
                name, dfs, summary = future.result()
                results_by_archetype[name] = dfs
                summaries.append(summary)
                print(f"[OK] Simulazione completata per archetipo: {name}")
            except Exception as e:
                print(f"[ERR] Simulazione fallita per archetipo {arch_name}: {e}")

    # Costruisco un DataFrame di sintesi
    if summaries:
        summary_df = pd.DataFrame(summaries).set_index("name")
    else:
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
# Esempio di utilizzo
# ---------------------------

if __name__ == "__main__":
    # Esempio: simula tutti gli archetipi definiti in BUILDING_ARCHETYPES
    # Se vuoi solo alcuni, filtra la lista:
    # archetypes_to_run = [a for a in BUILDING_ARCHETYPES if a["country"] == "Italy"]
    archetypes_to_run = BUILDING_ARCHETYPES

    results, summary_df = simulate_archetypes_in_parallel(
        archetypes_to_run,
        max_workers=None,  # o un numero fisso, es. 4
    )

    # Stampa riepilogo
    print("\n===== RIEPILOGO SINTESI =====")
    print(summary_df)

    # Esempio: accesso ai DataFrame di un singolo archetipo
    example_name = next(iter(results)) if results else None
    if example_name:
        df_hourly = results[example_name]["hourly_building"]
        df_annual = results[example_name]["annual_building"]
        df_system = results[example_name]["hourly_system"]

        print(f"\nEsempio risultati per: {example_name}")
        print("hourly_building:", df_hourly.shape)
        print("annual_building:", df_annual.shape)
        print("hourly_system:", df_system.shape)

        # Se vuoi, puoi anche salvare su CSV/parquet:
        # df_hourly.to_csv(f"hourly_{example_name}.csv", index=False)
        # df_annual.to_csv(f"annual_{example_name}.csv", index=False)
        # df_system.to_csv(f"system_{example_name}.csv", index=False)
