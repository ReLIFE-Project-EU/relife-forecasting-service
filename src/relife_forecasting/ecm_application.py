"""
ECM Multiprocessing con Debug e Gestione Errori
Versione che:
1. Importa correttamente la funzione di simulazione
2. Aggiunge logging dettagliato
3. Gestisce errori e timeout
4. Mostra progress bar
5. Salva ogni BUI modificato nella cartella building_examples
"""

import copy
import itertools
import os
import time
import traceback
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, List, Optional,Tuple
from pathlib import Path
import json
import numpy as np
import pybuildingenergy as pybui
from building_examples import BUI, INPUT_SYSTEM_HVAC
import pandas as pd

RESULTS_DIR = "results/ecm/"               
SYSTEM_RESULTS_DIR = "results/ecm/heating" 
BUILDING_EXAMPLES_DIR = "building_examples_ecm"

# ============================================================================
# IMPORTA LA TUA FUNZIONE DI SIMULAZIONE
# ============================================================================
# def simulate_hourly(BUI, INPUT_SYSTEM_HVAC, epw_name=None, name_file=None):
#     """Wrapper per pybuildingenergy"""
#     hourly_sim, annual_results = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
#         BUI,
#         weather_source="pvgis",
#         path_weather_file=epw_name,
#     )

#     # Salva risultati
#     hourly_sim.to_csv(name_file, index=True)

#     # Se vuoi anche i risultati annuali:
#     # annual_file = name_file.replace(".csv", "_annual.csv")
#     # annual_results.to_csv(annual_file, index=True)

#     return hourly_sim



def simulate_hourly(
    BUI: Dict[str, Any],
    INPUT_SYSTEM_HVAC: Dict[str, Any],
    epw_name: Optional[str] = None,
    name_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Wrapper per pybuildingenergy:
      - Calcola i fabbisogni edificio (ISO 52016)
      - Calcola il funzionamento del sistema di riscaldamento (ISO 15316)
      - Opzionalmente salva i risultati su file CSV.

    Args:
        BUI: Dizionario BUI in formato interno pybuildingenergy
        INPUT_SYSTEM_HVAC: Configurazione sistema di riscaldamento
        epw_name: path del file EPW (se fornito → usa weather_source='epw')
        name_file: path del CSV per i risultati edificio

    Returns:
        hourly_sim:     DataFrame orario edificio (ISO 52016)
        df_system:      DataFrame orario sistema (ISO 15316)
        annual_results: DataFrame risultati annuali edificio
    """

    # ===================== ISO 52016 – EDIFICIO =====================
    if epw_name:
        hourly_sim, annual_results = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
            BUI,
            weather_source="epw",
            path_weather_file=epw_name,
        )
    else:
        hourly_sim, annual_results = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
            BUI,
            weather_source="pvgis",
        )

    # ===================== ISO 15316 – SISTEMA =====================
    calc = pybui.HeatingSystemCalculator(INPUT_SYSTEM_HVAC)
    calc.load_csv_data(hourly_sim)
    df_system = calc.run_timeseries()

    # ===================== SALVATAGGIO CSV (OPZIONALE) =====================
    if name_file is not None:
        # 1) CSV edificio (nella cartella risultati "normale")
        hourly_sim.to_csv(name_file, index=True)

        # 2) CSV sistema in cartella dedicata (es. results/ecm/heating)
        system_dir = Path(SYSTEM_RESULTS_DIR)
        system_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(name_file).name                # es: test-cy__baseline__... .csv
        system_file = system_dir / base_name.replace(".csv", "_system.csv")

        df_system.to_csv(system_file, index=True)

        # opzionale: annuale
        # annual_file = Path(name_file).with_name(Path(name_file).stem + "_annual.csv")
        # annual_results.to_csv(annual_file, index=True)

    return hourly_sim, df_system, annual_results



# Importa BUI e INPUT_SYSTEM_HVAC
try:
    from building_examples import BUI, INPUT_SYSTEM_HVAC
except ImportError:
    print("⚠️  building_examples non trovato, usa dati di default")
    BUI = {"building": {"name": "test_building"}}
    INPUT_SYSTEM_HVAC = {}





# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def slugify(text: str) -> str:
    """Nome file safe per l'OS."""
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def build_name_file(
    building_name: str,
    ecm_combo: List[str],
    epw_name: str,
    base_dir: Optional[str] = None,
) -> str:
    """Costruisce nome file univoco per i risultati CSV."""
    dir_path = Path(base_dir or RESULTS_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)

    bld = slugify(building_name)
    ecm_tag = "_".join(sorted(ecm_combo)) if ecm_combo else "baseline"
    epw_tag = slugify(epw_name.split(".")[0])

    filename = f"{bld}__{ecm_tag}__{epw_tag}.csv"
    return str(dir_path / filename)


def save_bui_to_folder(
    BUI_obj: Dict[str, Any],
    active_elements: List[str],
    folder: str = BUILDING_EXAMPLES_DIR,
) -> str:
    """
    Salva il BUI modificato nella cartella `building_examples` come JSON.
    Il nome del file include il nome dell'edificio e la combinazione ECM.
    Esempio: BUI_test_building__wall_window.json
    """
    Path(folder).mkdir(parents=True, exist_ok=True)

    building_name = BUI_obj.get("building", {}).get("name", "building")
    combo_tag = "_".join(sorted(active_elements)) if active_elements else "baseline"

    filename = f"BUI_{slugify(building_name)}__{combo_tag}.json"
    full_path = Path(folder) / filename

    def _json_safe(obj: Any):
        """Converte numpy/array in tipi serializzabili JSON."""
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return [_json_safe(v) for v in obj.tolist()]
        return obj

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(BUI_obj), f, ensure_ascii=False, indent=2)

    print(f"  💾 BUI modificato salvato in: {full_path}")
    return str(full_path)


# ============================================================================
# CLASSIFICAZIONE SUPERFICI
# ============================================================================

def classify_surface(surface: Dict[str, Any]) -> Optional[str]:
    """Classifica superficie come 'roof', 'wall' o 'window'."""
    s_type = str(surface.get("type", "")).lower()
    ori = surface.get("orientation", {})
    tilt = float(ori.get("tilt", 0))
    azimuth = float(ori.get("azimuth", 0))
    name = str(surface.get("name", "")).lower()
    svf = float(surface.get("sky_view_factor", 0.0))

    # Tetto: opaque, orizzontale, verso cielo, non slab
    if (
        s_type == "opaque"
        and abs(tilt - 0) < 1e-3
        and abs(azimuth - 0) < 1e-3
        and svf > 0.01
        and "slab" not in name
    ):
        return "roof"

    # Muri: opachi verticali
    if s_type == "opaque" and abs(tilt - 90) < 1e-3:
        return "wall"

    # Finestre: trasparenti verticali
    if s_type == "transparent" and abs(tilt - 90) < 1e-3:
        return "window"

    return None


def apply_u_values_to_BUI(
    BUI_base: Dict[str, Any],
    u_wall: Optional[float],
    u_roof: Optional[float],
    u_window: Optional[float],
    active_elements: List[str],
) -> Dict[str, Any]:
    """
    Applica U-values agli elementi specificati e
    SALVA il BUI modificato in building_examples.
    """
    u_map: Dict[str, float] = {}

    if "roof" in active_elements and u_roof is not None:
        u_map["roof"] = u_roof
    if "wall" in active_elements and u_wall is not None:
        u_map["wall"] = u_wall
    if "window" in active_elements and u_window is not None:
        u_map["window"] = u_window

    BUI_new = copy.deepcopy(BUI_base)

    modified_count = 0
    for surface in BUI_new.get("building_surface", []):
        s_class = classify_surface(surface)
        if s_class in u_map:
            old_u = surface.get("u_value", "N/A")
            surface["u_value"] = u_map[s_class]
            modified_count += 1
            print(f"  ✓ {surface.get('name', 'Unknown')}: {s_class} U={old_u} → {u_map[s_class]}")

    print(f"  📝 Modificate {modified_count} superfici")

    # 🔴 QUI salviamo il BUI modificato nella cartella building_examples
    save_bui_to_folder(BUI_new, active_elements=active_elements)

    return BUI_new


# ============================================================================
# GENERAZIONE COMBINAZIONI ECM
# ============================================================================

def generate_ecm_combinations(ecm_options: List[str]) -> List[List[str]]:
    """Genera tutte le combinazioni NON VUOTE delle ECM."""
    combos: List[List[str]] = []
    for r in range(1, len(ecm_options) + 1):
        for subset in itertools.combinations(ecm_options, r):
            combos.append(list(subset))
    return combos


# ============================================================================
# SIMULAZIONE SINGOLA
# ============================================================================

def run_single_simulation(
    BUI: Dict[str, Any],
    INPUT_SYSTEM_HVAC: Dict[str, Any],
    name_file: str,
    epw_name: str,
) -> Dict[str, Any]:
    """
    Esegue una singola simulazione con gestione errori.
    Ritorna dict con status e info.
    """
    start_time = time.time()

    try:
        print(f"\n{'='*80}")
        print(f"   SIMULAZIONE: {Path(name_file).name}")
        print(f"   Building: {BUI['building']['name']}")
        print(f"   EPW: {epw_name}")
        print(f"{'='*80}")

        # Esegui simulazione
        result = simulate_hourly(
            BUI=BUI,
            INPUT_SYSTEM_HVAC=INPUT_SYSTEM_HVAC,
            epw_name=epw_name,
            name_file=name_file,
        )

        elapsed = time.time() - start_time

        # Verifica file creato
        if not os.path.exists(name_file):
            raise FileNotFoundError(f"File non creato: {name_file}")

        file_size = os.path.getsize(name_file) / 1024  # KB

        print(f"\n✅ SUCCESSO in {elapsed:.1f}s")
        print(f"   File: {name_file}")
        print(f"   Size: {file_size:.1f} KB")

        return {
            "status": "success",
            "file": name_file,
            "elapsed": elapsed,
            "size_kb": file_size,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"

        print(f"\n❌ ERRORE in {elapsed:.1f}s")
        print(f"   {error_msg}")
        print(f"   File: {name_file}")
        print(f"\n{traceback.format_exc()}")

        return {
            "status": "error",
            "file": name_file,
            "elapsed": elapsed,
            "error": error_msg,
        }


# Wrapper per multiprocessing
def _mp_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker per multiprocessing."""
    return run_single_simulation(
        BUI=task["BUI"],
        INPUT_SYSTEM_HVAC=task["INPUT_SYSTEM_HVAC"],
        name_file=task["name_file"],
        epw_name=task["epw_name"],
    )


# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def run_ecm_multiprocessing(
    BUI_base: Dict[str, Any],
    INPUT_SYSTEM_HVAC: Dict[str, Any],
    ecm_options: List[str],
    u_wall: Optional[float] = None,
    u_roof: Optional[float] = None,
    u_window: Optional[float] = None,
    epw_name: str = "GRC_Athens.167160_IWEC",
    n_processes: Optional[int] = None,
    output_dir: Optional[str] = None,
    include_baseline: bool = True,
) -> Dict[str, Any]:
    """
    Esegue simulazioni ECM in multiprocessing.
    
    Returns:
        Dict con statistiche delle simulazioni
    """

    print("\n" + "="*80)
    print("AVVIO ECM MULTIPROCESSING")
    print("="*80)

    # Validazioni
    if "wall" in ecm_options and u_wall is None:
        raise ValueError("Per ECM 'wall' è necessario fornire u_wall.")
    if "roof" in ecm_options and u_roof is None:
        raise ValueError("Per ECM 'roof' è necessario fornire u_roof.")
    if "window" in ecm_options and u_window is None:
        raise ValueError("Per ECM 'window' è necessario fornire u_window.")

    # Genera combinazioni
    ecm_combos = generate_ecm_combinations(ecm_options)

    # Aggiungi baseline se richiesto
    if include_baseline:
        ecm_combos.insert(0, [])  # Lista vuota = baseline

    print(f"\n === CONFIGURAZIONE: ===")
    print(f"   ECM options: {ecm_options}")
    print(f"   U-values: wall={u_wall}, roof={u_roof}, window={u_window}")
    print(f"   Combinazioni da simulare: {len(ecm_combos)}")
    print(f"   Processi paralleli: {n_processes or cpu_count()}")
    print(f"   Output dir: {output_dir or RESULTS_DIR}")

    base_dir = output_dir or RESULTS_DIR
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    # Prepara tasks
    tasks: List[Dict[str, Any]] = []

    print(f"\n📋 PREPARAZIONE TASKS:")
    for i, combo in enumerate(ecm_combos, 1):
        combo_name = ", ".join(combo) if combo else "BASELINE"
        print(f"\n[{i}/{len(ecm_combos)}] Combinazione: {combo_name}")

        name_file = build_name_file(
            building_name=BUI_base["building"]["name"],
            ecm_combo=combo,
            epw_name=epw_name,
            base_dir=base_dir,
        )

        print(f"   Output: {Path(name_file).name}")

        # Applica modifiche BUI
        if combo:  # Non modificare baseline
            BUI_mod = apply_u_values_to_BUI(
                BUI_base=BUI_base,
                u_wall=u_wall,
                u_roof=u_roof,
                u_window=u_window,
                active_elements=combo,
            )
        else:
            BUI_mod = copy.deepcopy(BUI_base)
            print("   (Baseline - nessuna modifica)")

        task = {
            "BUI": BUI_mod,
            "INPUT_SYSTEM_HVAC": INPUT_SYSTEM_HVAC,
            "name_file": name_file,
            "epw_name": epw_name,
        }
        tasks.append(task)

    # Esegui simulazioni
    print(f"\n{'='*80}")
    print(f"🔄 ESECUZIONE SIMULAZIONI ({len(tasks)} tasks)")
    print(f"{'='*80}")

    start_time = time.time()

    if n_processes is None or n_processes == 1:
        # Sequenziale (per debug)
        print("⚠️  Modalità SEQUENZIALE (n_processes=1)")
        results = [_mp_worker(task) for task in tasks]
    else:
        # Parallelo
        print(f"⚡ Modalità PARALLELA ({n_processes} processi)")
        with Pool(processes=n_processes) as pool:
            results = pool.map(_mp_worker, tasks)

    total_time = time.time() - start_time

    # Statistiche
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"\n{'='*80}")
    print(f"RISULTATI FINALI")
    print(f"{'='*80}")
    print(f"Successi: {len(successful)}/{len(results)}")
    print(f"Errori: {len(failed)}/{len(results)}")
    print(f"Tempo totale: {total_time:.1f}s")
    print(f"⚡ Tempo medio: {total_time/len(results):.1f}s per simulazione")

    if successful:
        total_size = sum(r["size_kb"] for r in successful)
        print(f"💾 Spazio disco: {total_size:.1f} KB totali")
        print(f"\n📁 File creati in: {base_dir}/")
        for r in successful:
            print(f"   ✓ {Path(r['file']).name}")

    if failed:
        print(f"\n⚠️  ERRORI:")
        for r in failed:
            print(f"   ✗ {Path(r['file']).name}: {r['error']}")

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_time": total_time,
        "results": results,
    }


# ============================================================================
# ESEMPIO D'USO
# ============================================================================

# if __name__ == "__main__":
    # Test con 1 solo processo per debug
stats = run_ecm_multiprocessing(
    BUI_base=BUI,
    INPUT_SYSTEM_HVAC=INPUT_SYSTEM_HVAC,
    ecm_options=["wall", "window"],
    u_wall=0.5,
    u_window=1.0,
    epw_name="/Users/dantonucci/Documents/GitHub/relife-forecasting-service/src/relife_forecasting/utils/GRC_Athens.167160_IWEC.epw",
    output_dir="results",
    n_processes=1,  # ← IMPORTANTE: usa 1 per debug!
    include_baseline=True,
)

print("\n" + "="*80)
print("COMPLETATO")
print("="*80)
print(f"Successi: {stats['successful']}/{stats['total']}")
print(f"Tempo: {stats['total_time']:.1f}s")
