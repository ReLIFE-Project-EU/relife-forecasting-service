"""The building archetypes used to live in this file. The API still imports
BUILDING_ARCHETYPES from here, and some analysis scripts load this file directly
by path. Moving or deleting it would break those callers.

Both ways of loading the file use the same catalogue. The code below also makes
each building and its UNI11300 settings available under their old names.

Loading a package by hand and writing names into globals() are code smells.
We accept them here because older scripts load this file by path and callers
can look up names like "AT_SFH_1946-1969_BUILDING". Keep these workarounds here.

To edit an archetype, look in catalogue/records. To change how its simulation
inputs are calculated, look in catalogue/builders.py.
"""

from __future__ import annotations

import sys
from importlib import util
from pathlib import Path
from typing import Any
from uuid import uuid4


def _load_sibling_catalogue():
    """Load the adjacent package for standalone imports and runpy callers."""
    directory = Path(__file__).resolve().with_name("catalogue")
    name = "_relife_catalogue_" + uuid4().hex
    spec = util.spec_from_file_location(name, directory / "__init__.py")
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    # Give the folder a temporary package name so its files can import each other.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        # Remove that name and its loaded modules so the next call gets a fresh catalogue.
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


if __package__:
    # The API imports relife_forecasting.building_examples. Python knows the
    # package name, so the dot means "look for catalogue inside relife_forecasting".
    from .catalogue import builders, records
else:
    # Older scripts load building_examples on its own. Python has no package
    # name for the dot, so we find the catalogue folder next to this file.
    _catalogue = _load_sibling_catalogue()
    builders = _catalogue.builders
    records = _catalogue.records

UNI11300_SIMULATION_EXAMPLE = builders.UNI11300_SIMULATION_EXAMPLE
SYSTEM_CONDENSING_BOILER_RADIATORS = builders.SYSTEM_CONDENSING_BOILER_RADIATORS
_HDD_BY_COUNTRY = builders._HDD_BY_COUNTRY
_EXTENDED_ARCHETYPE_SPECS = records.source_records()

BUILDING_ARCHETYPES = [
    builders.build_archetype(name, spec) for name, spec in _EXTENDED_ARCHETYPE_SPECS.items()
]

# Publish the names used by legacy callers, including names with hyphens.
globals().update(
    {
        f"{entry['name']}_{suffix}": entry[field]
        for entry in BUILDING_ARCHETYPES
        for suffix, field in (("BUILDING", "bui"), ("UNI11300", "uni11300"))
    }
)

BUI_SINGLE_FAMILY_1946_1969 = next(
    entry["bui"] for entry in BUILDING_ARCHETYPES if entry["name"] == "AT_SFH_1946-1969"
)


def build_iso52016_uni11300_pv_payload(
    archetype: dict[str, Any],
    pv: dict[str, Any],
    return_hourly_building: bool = False,
) -> dict[str, Any]:
    """
    Build a normalized request body for /run/iso52016-uni11300-pv from an archetype.

    The returned payload has the exact shape expected by the endpoint:
      {
        "bui": {...},
        "pv": {...},
        "uni11300": {"input_unit", "heating_params", "cooling_params"},
        "return_hourly_building": bool
      }
    """
    uni11300_cfg = (
        archetype.get("uni11300")
        or archetype.get("systems_archetype")
        or UNI11300_SIMULATION_EXAMPLE
    )

    return {
        "bui": archetype["bui"],
        "pv": pv,
        "uni11300": uni11300_cfg,
        "return_hourly_building": bool(return_hourly_building),
    }
