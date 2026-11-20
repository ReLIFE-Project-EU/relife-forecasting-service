"""Adapt workbook thermal capacities to the convention selected for the catalogue."""

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
    """Select the workbook convention once, when the catalogue builder is defined."""
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
            # The solver ignores these workbook classes. Use evenly distributed mass.
            prepared["construction_class"] = ConstructionClass.DISTRIBUTED.value
        elif source_class != ConstructionClass.I:
            raise ValueError(f"Unknown extended construction class: {source_class}")

        prepared["roof_thermal_capacity"] /= spec["roof_area"]
        # Wall capacity covers the gross area, including window openings.
        prepared["wall_thermal_capacity"] /= spec["wall_area"]
        prepared["slab_thermal_capacity"] /= spec["slab_area"]

        entry = build(name, prepared)
        entry["bui"]["units"]["thermal_capacity"] = ThermalCapacityUnit.AREAL.value
        return entry

    return build_provisional
