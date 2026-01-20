"""
ECM multiprocessing with debug and error handling.
Version that:
1. Imports the simulation function correctly
2. Adds detailed logging
3. Handles errors and timeouts
4. Shows a progress bar
5. Saves each modified BUI into the building_examples folder
"""

import copy
import itertools
import os
import time
import traceback
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pybuildingenergy as pybui
import pandas as pd
from typing import Union

# Import BUI and INPUT_SYSTEM_HVAC
try:
    from building_examples import (
        BUI_SINGLE_FAMILY_1946_1969,
        INPUT_SYSTEM_HVAC_CONDENSING_BOILER_AND_RADIATOR,
    )
except ImportError:
    print("[WARN] building_examples not found, using default data")
    BUI = {"building": {"name": "test_building"}}
    INPUT_SYSTEM_HVAC = {}

BUI = BUI_SINGLE_FAMILY_1946_1969
RESULTS_DIR = "results/ecm/"
SYSTEM_RESULTS_DIR = "results/ecm/heating"
BUILDING_EXAMPLES_DIR = "building_examples_ecm"

# ============================================================================
# IMPORT YOUR SIMULATION FUNCTION
# ============================================================================
# def simulate_hourly(BUI, INPUT_SYSTEM_HVAC, epw_name=None, name_file=None):
#     """Wrapper for pybuildingenergy"""
#     hourly_sim, annual_results = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
#         BUI,
#         weather_source="pvgis",
#         path_weather_file=epw_name,
#     )
#
#     # Save results
#     hourly_sim.to_csv(name_file, index=True)
#
#     # If you also want annual results:
#     # annual_file = name_file.replace(".csv", "_annual.csv")
#     # annual_results.to_csv(annual_file, index=True)
#
#     return hourly_sim


def simulate_hourly(
    BUI: Dict[str, Any],
    INPUT_SYSTEM_HVAC: Dict[str, Any],
    epw_name: Optional[str] = None,
    name_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Wrapper for pybuildingenergy:
      - Computes building needs (ISO 52016)
      - Computes heating system operation (ISO 15316)
      - Optionally saves results to CSV.

    Args:
        BUI: BUI dictionary in pybuildingenergy internal format
        INPUT_SYSTEM_HVAC: Heating system configuration
        epw_name: EPW file path (if provided -> use weather_source='epw')
        name_file: CSV path for building results

    Returns:
        hourly_sim:     Hourly building DataFrame (ISO 52016)
        df_system:      Hourly system DataFrame (ISO 15316)
        annual_results: Annual building results DataFrame
    """

    # ===================== ISO 52016 - BUILDING =====================
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

    # ===================== ISO 15316 - SYSTEM =====================
    calc = pybui.HeatingSystemCalculator(INPUT_SYSTEM_HVAC)
    calc.load_csv_data(hourly_sim)
    df_system = calc.run_timeseries()

    # ===================== CSV SAVE (OPTIONAL) =====================
    if name_file is not None:
        # 1) Building CSV (in the main results folder)
        hourly_sim.to_csv(name_file, index=True)

        # 2) System CSV in a dedicated folder (e.g., results/ecm/heating)
        system_dir = Path(SYSTEM_RESULTS_DIR)
        system_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(name_file).name  # e.g., test-cy__baseline__... .csv
        system_file = system_dir / base_name.replace(".csv", "_system.csv")

        df_system.to_csv(system_file, index=True)

        # Optional: annual results
        # annual_file = Path(name_file).with_name(Path(name_file).stem + "_annual.csv")
        # annual_results.to_csv(annual_file, index=True)

    return hourly_sim, df_system, annual_results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def slugify(text: str) -> str:
    """OS-safe filename."""
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
    epw_name: Union[str, Path],
    base_dir: Optional[str] = None,
) -> str:
    """Build a unique filename for CSV results."""
    dir_path = Path(base_dir or RESULTS_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)

    bld = slugify(building_name)
    ecm_tag = "_".join(sorted(ecm_combo)) if ecm_combo else "baseline"

    epw_path = Path(epw_name)  # works for str or Path
    epw_tag = slugify(epw_path.stem)  # filename without extension

    filename = f"{bld}__{ecm_tag}__{epw_tag}.csv"
    return str(dir_path / filename)


def save_bui_to_folder(
    BUI_obj: Dict[str, Any],
    active_elements: List[str],
    folder: str = BUILDING_EXAMPLES_DIR,
) -> str:
    """
    Save the modified BUI into the `building_examples` folder as JSON.
    The filename includes the building name and the ECM combination.
    Example: BUI_test_building__wall_window.json
    """
    Path(folder).mkdir(parents=True, exist_ok=True)

    building_name = BUI_obj.get("building", {}).get("name", "building")
    combo_tag = "_".join(sorted(active_elements)) if active_elements else "baseline"

    filename = f"BUI_{slugify(building_name)}__{combo_tag}.json"
    full_path = Path(folder) / filename

    def _json_safe(obj: Any):
        """Convert numpy/arrays into JSON-serializable types."""
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

    print(f"  [SAVE] Modified BUI saved to: {full_path}")
    return str(full_path)


# ============================================================================
# SURFACE CLASSIFICATION
# ============================================================================

def classify_surface(surface: Dict[str, Any]) -> Optional[str]:
    """Classify surface as 'roof', 'wall', or 'window'."""
    s_type = str(surface.get("type", "")).lower()
    ori = surface.get("orientation", {})
    tilt = float(ori.get("tilt", 0))
    azimuth = float(ori.get("azimuth", 0))
    name = str(surface.get("name", "")).lower()
    svf = float(surface.get("sky_view_factor", 0.0))

    # Roof: opaque, horizontal, facing sky, not a slab
    if (
        s_type == "opaque"
        and abs(tilt - 0) < 1e-3
        and abs(azimuth - 0) < 1e-3
        and svf > 0.01
        and "slab" not in name
    ):
        return "roof"

    # Walls: opaque vertical
    if s_type == "opaque" and abs(tilt - 90) < 1e-3:
        return "wall"

    # Windows: transparent vertical
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
    Apply U-values to the specified elements and
    SAVE the modified BUI in building_examples.
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
            print(
                f"  [OK] {surface.get('name', 'Unknown')}: "
                f"{s_class} U={old_u} -> {u_map[s_class]}"
            )

    print(f"  [INFO] Modified surfaces: {modified_count}")

    # Save the modified BUI into building_examples
    save_bui_to_folder(BUI_new, active_elements=active_elements)

    return BUI_new


# ============================================================================
# ECM COMBINATION GENERATION
# ============================================================================

def generate_ecm_combinations(ecm_options: List[str]) -> List[List[str]]:
    """Generate all NON-EMPTY ECM combinations."""
    combos: List[List[str]] = []
    for r in range(1, len(ecm_options) + 1):
        for subset in itertools.combinations(ecm_options, r):
            combos.append(list(subset))
    return combos


# ============================================================================
# SINGLE SIMULATION
# ============================================================================

def run_single_simulation(
    BUI: Dict[str, Any],
    INPUT_SYSTEM_HVAC: Dict[str, Any],
    name_file: str,
    epw_name: str,
) -> Dict[str, Any]:
    """
    Run a single simulation with error handling.
    Returns a dict with status and info.
    """
    start_time = time.time()

    try:
        print(f"\n{'='*80}")
        print(f"   SIMULATION: {Path(name_file).name}")
        print(f"   Building: {BUI['building']['name']}")
        print(f"   EPW: {epw_name}")
        print(f"{'='*80}")

        # Run simulation
        result = simulate_hourly(
            BUI=BUI,
            INPUT_SYSTEM_HVAC=INPUT_SYSTEM_HVAC,
            epw_name=epw_name,
            name_file=name_file,
        )

        elapsed = time.time() - start_time

        # Check file creation
        if not os.path.exists(name_file):
            raise FileNotFoundError(f"File not created: {name_file}")

        file_size = os.path.getsize(name_file) / 1024  # KB

        print(f"\n[OK] SUCCESS in {elapsed:.1f}s")
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

        print(f"\n[ERROR] FAILED in {elapsed:.1f}s")
        print(f"   {error_msg}")
        print(f"   File: {name_file}")
        print(f"\n{traceback.format_exc()}")

        return {
            "status": "error",
            "file": name_file,
            "elapsed": elapsed,
            "error": error_msg,
        }


# Multiprocessing worker
def _mp_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker for multiprocessing."""
    return run_single_simulation(
        BUI=task["BUI"],
        INPUT_SYSTEM_HVAC=task["INPUT_SYSTEM_HVAC"],
        name_file=task["name_file"],
        epw_name=task["epw_name"],
    )


# ============================================================================
# MAIN FUNCTION
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
    Run ECM simulations with multiprocessing.

    Returns:
        Dict with simulation statistics
    """

    print("\n" + "=" * 80)
    print("START ECM MULTIPROCESSING")
    print("=" * 80)

    # Validations
    if "wall" in ecm_options and u_wall is None:
        raise ValueError("For ECM 'wall' you must provide u_wall.")
    if "roof" in ecm_options and u_roof is None:
        raise ValueError("For ECM 'roof' you must provide u_roof.")
    if "window" in ecm_options and u_window is None:
        raise ValueError("For ECM 'window' you must provide u_window.")

    # Generate combinations
    ecm_combos = generate_ecm_combinations(ecm_options)

    # Add baseline if requested
    if include_baseline:
        ecm_combos.insert(0, [])  # Empty list = baseline

    print("\n === CONFIGURATION: ===")
    print(f"   ECM options: {ecm_options}")
    print(f"   U-values: wall={u_wall}, roof={u_roof}, window={u_window}")
    print(f"   Combinations to simulate: {len(ecm_combos)}")
    print(f"   Parallel processes: {n_processes or cpu_count()}")
    print(f"   Output dir: {output_dir or RESULTS_DIR}")

    base_dir = output_dir or RESULTS_DIR
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    # Prepare tasks
    tasks: List[Dict[str, Any]] = []

    print("\n[INFO] PREPARING TASKS:")
    for i, combo in enumerate(ecm_combos, 1):
        combo_name = ", ".join(combo) if combo else "BASELINE"
        print(f"\n[{i}/{len(ecm_combos)}] Combination: {combo_name}")

        name_file = build_name_file(
            building_name=BUI_base["building"]["name"],
            ecm_combo=combo,
            epw_name=epw_name,
            base_dir=base_dir,
        )

        print(f"   Output: {Path(name_file).name}")

        # Apply BUI changes
        if combo:  # Do not modify baseline
            BUI_mod = apply_u_values_to_BUI(
                BUI_base=BUI_base,
                u_wall=u_wall,
                u_roof=u_roof,
                u_window=u_window,
                active_elements=combo,
            )
        else:
            BUI_mod = copy.deepcopy(BUI_base)
            print("   (Baseline - no changes)")

        task = {
            "BUI": BUI_mod,
            "INPUT_SYSTEM_HVAC": INPUT_SYSTEM_HVAC,
            "name_file": name_file,
            "epw_name": epw_name,
        }
        tasks.append(task)

    # Run simulations
    print(f"\n{'='*80}")
    print(f"[RUN] EXECUTING SIMULATIONS ({len(tasks)} tasks)")
    print(f"{'='*80}")

    start_time = time.time()

    if n_processes is None or n_processes == 1:
        # Sequential (for debug)
        print("[WARN] SEQUENTIAL MODE (n_processes=1)")
        results = [_mp_worker(task) for task in tasks]
    else:
        # Parallel
        print(f"[INFO] PARALLEL MODE ({n_processes} processes)")
        with Pool(processes=n_processes) as pool:
            results = pool.map(_mp_worker, tasks)

    total_time = time.time() - start_time

    # Stats
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Successes: {len(successful)}/{len(results)}")
    print(f"Errors: {len(failed)}/{len(results)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"[INFO] Avg time: {total_time/len(results):.1f}s per simulation")

    if successful:
        total_size = sum(r["size_kb"] for r in successful)
        print(f"[INFO] Disk usage: {total_size:.1f} KB total")
        print(f"\n[INFO] Files created in: {base_dir}/")
        for r in successful:
            print(f"   OK {Path(r['file']).name}")

    if failed:
        print("\n[WARN] ERRORS:")
        for r in failed:
            print(f"   FAIL {Path(r['file']).name}: {r['error']}")

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_time": total_time,
        "results": results,
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# if __name__ == "__main__":
# Test with a single process for debug
stats = run_ecm_multiprocessing(
    BUI_base=BUI,
    INPUT_SYSTEM_HVAC=INPUT_SYSTEM_HVAC_CONDENSING_BOILER_AND_RADIATOR,
    ecm_options=["wall", "window"],
    u_wall=0.5,
    u_window=1.0,
    epw_name=Path("epw_weather/GRC_Athens.167160_IWEC.epw"),
    output_dir="results",
    n_processes=1,  # IMPORTANT: use 1 for debug
    include_baseline=True,
)

print("\n" + "=" * 80)
print("COMPLETED")
print("=" * 80)
print(f"Successes: {stats['successful']}/{stats['total']}")
print(f"Time: {stats['total_time']:.1f}s")
