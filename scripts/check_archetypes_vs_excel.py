"""
Check the extended country archetypes in building_examples.py against the Excel
workbooks in "EU_new archetypes/" (one sheet per archetype, e.g. AT_SFH_1946-1969).

  python scripts/check_archetypes_vs_excel.py
Exit code 1 if any value differs.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import openpyxl

from relife_forecasting import building_examples as be

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {  # spec key -> (group, parameter) in the sheet
    "n_floors": ("Geometry", "Floors"),
    "exposed_perimeter": ("Geometry", "Perimeter (external, estimated)"),
    "roof_area": ("Geometry", "Building footprint / ground area"),
    "slab_area": ("Geometry", "Building footprint / ground area"),
    "wall_area": ("Geometry", "External wall area (gross)"),
    "window_area": ("Geometry", "Window area"),
    "wall_u": ("Envelope", "External wall U-value"),
    "roof_u": ("Envelope", "Roof U-value"),
    "slab_u": ("Envelope", "Floor/Slab U-value"),
    "window_u": ("Windows", "Window Uw (whole window)"),
    "g_value": ("Windows", "g-value"),
    "heating_eta_generation": ("Heating system", "COP / seasonal efficiency"),
    "cooling_cop_generation": ("Cooling system", "SEER"),
    "dhw_eta_generation": ("DHW", "Generation efficiency_gen,DHW"),
}


def main() -> int:
    specs = be._EXTENDED_ARCHETYPE_SPECS
    problems = []
    seen = set()
    for path in sorted(glob.glob(str(ROOT / "EU_new archetypes" / "[!~]*.xlsx"))):
        for ws in openpyxl.load_workbook(path, data_only=True):
            seen.add(ws.title)
            rows = {(r[0], r[1]): r[2] for r in ws.iter_rows(min_row=2, values_only=True) if r[0] and r[1]}
            spec = specs.get(ws.title)
            if spec is None:
                problems.append(f"{ws.title}: sheet has no spec in the code")
                continue
            for key, label in FIELDS.items():
                expected, actual = rows.get(label), spec.get(key)
                if expected is None or actual is None or abs(float(actual) - float(expected)) > 1e-3 * max(1, abs(float(expected))):
                    problems.append(f"{ws.title}: {key} code={actual} sheet={expected}")
    problems += [f"{n}: spec has no sheet" for n in specs if n not in seen]
    print(f"{len(seen)} sheets, {len(specs)} specs, {len(problems)} differences")
    print("\n".join(problems[:50]))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
