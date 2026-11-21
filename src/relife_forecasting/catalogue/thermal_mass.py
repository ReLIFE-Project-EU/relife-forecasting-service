"""Prepare catalogue heat capacities for pybuildingenergy.

The solver expects heat capacity per square metre. Provisional mode treats
the workbook values as totals (J/K) for the roof, the ground slab and all
walls combined. It divides each total by its area and maps two construction
classes the solver ignores.

Set RELIFE_EXTENDED_THERMAL_MASS_MODE=provisional before startup to use this
interpretation. The default, legacy, preserves the inputs used by existing simulations.
"""

import os
from enum import StrEnum
from functools import wraps
from typing import Any, Protocol

from .spec import ArchetypeSpec, ConstructionClass


class ThermalCapacityUnit(StrEnum):
    TOTAL = "J/K"
    AREAL = "J/(m²·K)"


class _ThermalMassMode(StrEnum):
    LEGACY = "legacy"
    PROVISIONAL = "provisional"


class _ArchetypeBuilder(Protocol):
    def __call__(self, name: str, spec: ArchetypeSpec) -> dict[str, Any]: ...


def with_provisional_thermal_mass(build: _ArchetypeBuilder) -> _ArchetypeBuilder:
    """Convert a copy for each provisional build so the workbook totals can be reused."""
    setting = "RELIFE_EXTENDED_THERMAL_MASS_MODE"
    try:
        mode = _ThermalMassMode(os.getenv(setting, _ThermalMassMode.LEGACY).strip().lower())
    except ValueError as exc:
        choices = " or ".join(repr(mode.value) for mode in _ThermalMassMode)
        raise ValueError(f"{setting} must be {choices}.") from exc
    if mode is _ThermalMassMode.LEGACY:
        return build

    @wraps(build)
    def build_provisional(name: str, spec: ArchetypeSpec) -> dict[str, Any]:
        prepared = spec.copy()
        source_class = spec["construction_class"]
        if source_class in (ConstructionClass.II, ConstructionClass.III):
            # With II or III the solver assigns zero heat capacity. Class D
            # distributes the supplied heat capacity through the construction.
            prepared["construction_class"] = ConstructionClass.DISTRIBUTED.value
        elif source_class != ConstructionClass.I:
            raise ValueError(f"Unknown extended construction class: {source_class}")

        prepared["roof_thermal_capacity"] /= spec["roof_area"]
        # The workbook's wall area includes window openings on all four façades.
        prepared["wall_thermal_capacity"] /= spec["wall_area"]
        prepared["slab_thermal_capacity"] /= spec["slab_area"]

        entry = build(name, prepared)
        entry["bui"]["units"]["thermal_capacity"] = ThermalCapacityUnit.AREAL.value
        return entry

    return build_provisional
