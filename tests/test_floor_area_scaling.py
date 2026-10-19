import copy

import pytest
from fastapi.testclient import TestClient

from relife_forecasting.building_examples import BUILDING_ARCHETYPES
from relife_forecasting.main import app
from relife_forecasting.routes.forecasting_service_functions import (
    classify_surface,
    scale_bui_floor_area,
)

client = TestClient(app)


def _sfh():
    return next(
        a for a in BUILDING_ARCHETYPES
        if a["country"] == "Austria" and a["name"] == "AT_SFH_1946-1969"
    )["bui"]


def _total(bui, kind):
    return sum(s["area"] for s in bui["building_surface"] if classify_surface(s) == kind)


def test_same_area_is_a_no_op_copy():
    bui = _sfh()
    out = scale_bui_floor_area(bui, bui["building"]["net_floor_area"])
    assert out == bui and out is not bui


def test_scaling_geometry():
    bui = _sfh()
    original = copy.deepcopy(bui)
    k = 1.5
    s = k ** 0.5
    out = scale_bui_floor_area(bui, bui["building"]["net_floor_area"] * k)

    assert bui == original  # input untouched
    assert out["building"]["net_floor_area"] == pytest.approx(bui["building"]["net_floor_area"] * k)
    assert out["building"]["exposed_perimeter"] == pytest.approx(bui["building"]["exposed_perimeter"] * s)
    assert out["building"]["height"] == bui["building"]["height"]
    assert _total(out, "roof") == pytest.approx(_total(bui, "roof") * k)
    assert _total(out, "slab") == pytest.approx(_total(bui, "slab") * k)
    assert _total(out, "wall") == pytest.approx(_total(bui, "wall") * s)
    assert _total(out, "window") == pytest.approx(_total(bui, "window") * s)
    cap0 = bui["building_parameters"]["system_capacities"]["heating_capacity"]
    assert out["building_parameters"]["system_capacities"]["heating_capacity"] == pytest.approx(cap0 * k)


def test_invalid_area_rejected():
    with pytest.raises(ValueError):
        scale_bui_floor_area(_sfh(), -5)


def test_building_endpoint_uses_floor_area():
    params = {
        "archetype": "true",
        "category": "Single Family House",
        "country": "Austria",
        "name": "AT_SFH_1946-1969",
    }
    base = client.post("/building", params=params).json()
    big = client.post("/building", params={**params, "floor_area": 250}).json()
    assert base["bui"]["building"]["net_floor_area"] == _sfh()["building"]["net_floor_area"]
    assert big["bui"]["building"]["net_floor_area"] == 250.0
    assert client.post("/building", params={**params, "floor_area": 0}).status_code == 422
