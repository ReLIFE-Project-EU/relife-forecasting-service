import runpy
from unittest.mock import Mock

import pytest

from relife_forecasting import building_examples
from relife_forecasting.catalogue.records.at import RECORDS
from relife_forecasting.catalogue.spec import ConstructionClass
from relife_forecasting.catalogue.thermal_mass import with_provisional_thermal_mass

SETTING = "RELIFE_EXTENDED_THERMAL_MASS_MODE"
ARCHETYPE = "AT_SFH_1946-1969"


@pytest.mark.parametrize(
    "source_class,expected_class",
    [
        (ConstructionClass.I, ConstructionClass.I),
        (ConstructionClass.II, ConstructionClass.DISTRIBUTED),
        (ConstructionClass.III, ConstructionClass.DISTRIBUTED),
    ],
)
def test_provisional_thermal_mass_prepares_workbook_inputs(
    monkeypatch, source_class, expected_class
):
    monkeypatch.setenv(SETTING, "provisional")
    spec = RECORDS[ARCHETYPE].copy()
    # Different roof and slab areas make a swapped divisor visible.
    spec.update(
        construction_class=source_class.value,
        roof_area=50.0,
        slab_area=40.0,
        wall_area=100.0,
        roof_thermal_capacity=100.0,
        slab_thermal_capacity=200.0,
        wall_thermal_capacity=300.0,
    )
    original = spec.copy()
    build = Mock(return_value={"bui": {"units": {}}})

    with_provisional_thermal_mass(build)(ARCHETYPE, spec)

    build.assert_called_once_with(
        ARCHETYPE,
        {
            **original,
            "construction_class": expected_class.value,
            "roof_thermal_capacity": 2.0,
            "slab_thermal_capacity": 5.0,
            "wall_thermal_capacity": 3.0,
        },
    )
    assert spec == original


@pytest.mark.parametrize(
    "setting,expected",
    [(None, ("class_iii", "J/K")), (" Provisional ", ("class_d", "J/(m²·K)")), ("typo", None)],
)
def test_catalogue_thermal_mass_at_startup(monkeypatch, setting, expected):
    """Load a fresh catalogue so the builder reads the setting for this case."""
    if setting is None:
        monkeypatch.delenv(SETTING, raising=False)
    else:
        monkeypatch.setenv(SETTING, setting)
    if expected is None:
        with pytest.raises(ValueError, match=SETTING):
            runpy.run_path(building_examples.__file__)
        return

    bui = runpy.run_path(building_examples.__file__)[ARCHETYPE + "_BUILDING"]
    assert (bui["building"]["construction_class"], bui["units"]["thermal_capacity"]) == expected
    spec = RECORDS[ARCHETYPE]
    wall_capacity = spec["wall_thermal_capacity"]
    if setting is not None:
        wall_capacity /= spec["wall_area"]
    walls = [s for s in bui["building_surface"] if s["name"].startswith("Opaque ")]
    assert [wall["thermal_capacity"] for wall in walls] == [wall_capacity] * 4
