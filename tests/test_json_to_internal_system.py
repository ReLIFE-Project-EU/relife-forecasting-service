import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from relife_forecasting.main import app
from relife_forecasting.routes.forecasting_service_functions import json_to_internal_system


def test_json_to_internal_system_accepts_list_of_records():
    system = {
        "gen_outdoor_temp_data": [{"outdoor_temp": 7.0, "supply_temp": 35.0}],
        "heat_emission_data": [
            {"value": 45.0},
            {"value": 10.0},
            {"value": 35.0},
            {"value": 0.6},
            {"value": 25.0},
        ],
        "outdoor_temp_data": [
            {"value": -5.0},
            {"value": 15.0},
            {"value": 45.0},
            {"value": 25.0},
        ],
    }

    converted = json_to_internal_system(system)

    assert isinstance(converted["gen_outdoor_temp_data"], pd.DataFrame)
    assert isinstance(converted["heat_emission_data"], pd.DataFrame)
    assert isinstance(converted["outdoor_temp_data"], pd.DataFrame)
    assert list(converted["gen_outdoor_temp_data"].index) == ["Generator curve"]
    assert list(converted["heat_emission_data"].index) == [
        "Max flow temperature HZ1",
        "Max Δθ flow / return HZ1",
        "Desired return temperature HZ1",
        "Desired load factor with ON-OFF for HZ1",
        "Minimum flow temperature for HZ1",
    ]
    assert list(converted["outdoor_temp_data"].index) == [
        "Minimum outdoor temperature",
        "Maximum outdoor temperature",
        "Maximum flow temperature",
        "Minimum flow temperature",
    ]


def test_json_to_internal_system_accepts_column_oriented_dict():
    system = {
        "gen_outdoor_temp_data": {"outdoor_temp": [7.0], "supply_temp": [35.0]},
        "heat_emission_data": {"value": [45.0, 10.0, 35.0, 0.6, 25.0]},
        "outdoor_temp_data": {"value": [-5.0, 15.0, 45.0, 25.0]},
    }

    converted = json_to_internal_system(system)

    assert list(converted["heat_emission_data"].index) == [
        "Max flow temperature HZ1",
        "Max Δθ flow / return HZ1",
        "Desired return temperature HZ1",
        "Desired load factor with ON-OFF for HZ1",
        "Minimum flow temperature for HZ1",
    ]
    assert list(converted["outdoor_temp_data"].index) == [
        "Minimum outdoor temperature",
        "Maximum outdoor temperature",
        "Maximum flow temperature",
        "Minimum flow temperature",
    ]


def test_json_to_internal_system_accepts_single_row_wide_legacy_tables():
    system = {
        "heat_emission_data": [
            {
                "theta_flow_max": 70.0,
                "delta_theta_max": 20.0,
                "theta_return_req": 45.0,
                "beta_req": 0.8,
                "theta_flow_min": 30.0,
            }
        ],
        "outdoor_temp_data": [
            {
                "theta_ext_min": -10.0,
                "theta_ext_max": 16.0,
                "theta_flow_max": 70.0,
                "theta_flow_min": 30.0,
            }
        ],
    }

    converted = json_to_internal_system(system)

    heat_df = converted["heat_emission_data"]
    outdoor_df = converted["outdoor_temp_data"]

    assert isinstance(heat_df, pd.DataFrame)
    assert isinstance(outdoor_df, pd.DataFrame)
    assert list(heat_df.index) == [
        "Max flow temperature HZ1",
        "Max Δθ flow / return HZ1",
        "Desired return temperature HZ1",
        "Desired load factor with ON-OFF for HZ1",
        "Minimum flow temperature for HZ1",
    ]
    assert list(outdoor_df.index) == [
        "Minimum outdoor temperature",
        "Maximum outdoor temperature",
        "Maximum flow temperature",
        "Minimum flow temperature",
    ]
    assert list(heat_df.columns) == [
        "theta_flow_max",
        "delta_theta_max",
        "theta_return_req",
        "beta_req",
        "theta_flow_min",
    ]
    assert list(outdoor_df.columns) == [
        "theta_ext_min",
        "theta_ext_max",
        "theta_flow_max",
        "theta_flow_min",
    ]
    assert len(heat_df) == 5
    assert len(outdoor_df) == 4


def test_json_to_internal_system_accepts_scalar_lists_with_canonical_columns():
    system = {
        "heat_emission_data": [70.0, 20.0, 45.0, 0.8, 30.0],
        "outdoor_temp_data": [-10.0, 16.0, 70.0, 30.0],
    }

    converted = json_to_internal_system(system)
    heat_df = converted["heat_emission_data"]
    outdoor_df = converted["outdoor_temp_data"]

    assert isinstance(heat_df, pd.DataFrame)
    assert isinstance(outdoor_df, pd.DataFrame)
    assert "θH_em_ret_req_sahz_i" in heat_df.columns
    assert "θext_min_sahz_i" in outdoor_df.columns
    assert list(heat_df.index) == [
        "Max flow temperature HZ1",
        "Max Δθ flow / return HZ1",
        "Desired return temperature HZ1",
        "Desired load factor with ON-OFF for HZ1",
        "Minimum flow temperature for HZ1",
    ]
    assert list(outdoor_df.index) == [
        "Minimum outdoor temperature",
        "Maximum outdoor temperature",
        "Maximum flow temperature",
        "Minimum flow temperature",
    ]


def test_json_to_internal_system_invalid_type_returns_422_and_logs(caplog):
    with caplog.at_level("WARNING"):
        with pytest.raises(HTTPException) as exc_info:
            json_to_internal_system({"heat_emission_data": None})

    exc = exc_info.value
    assert exc.status_code == 422
    assert "heat_emission_data" in str(exc.detail)
    assert "expected a JSON list or object" in str(exc.detail)
    assert "Invalid system payload for 'heat_emission_data'" in caplog.text


def test_json_to_internal_system_strict_row_count_returns_422_and_logs(caplog):
    with caplog.at_level("WARNING"):
        with pytest.raises(HTTPException) as exc_info:
            json_to_internal_system({"heat_emission_data": {"value": [45.0]}})

    exc = exc_info.value
    assert exc.status_code == 422
    assert "expected 5 rows" in str(exc.detail)
    assert "Invalid row count for 'heat_emission_data'" in caplog.text


def test_json_to_internal_system_gen_outdoor_preserves_backward_compatibility(caplog):
    with caplog.at_level("WARNING"):
        converted = json_to_internal_system(
            {
                "gen_outdoor_temp_data": {
                    "outdoor_temp": [7.0, 10.0],
                    "supply_temp": [35.0, 40.0],
                }
            }
        )

    df = converted["gen_outdoor_temp_data"]
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["Generator curve", "Generator curve"]
    assert "Unexpected row count for 'gen_outdoor_temp_data'" in caplog.text


def test_json_to_internal_system_dataframe_input_is_left_unchanged():
    input_df = pd.DataFrame({"value": [1.0, 2.0]})
    input_df.index = ["custom-1", "custom-2"]

    converted = json_to_internal_system({"heat_emission_data": input_df})

    out_df = converted["heat_emission_data"]
    assert isinstance(out_df, pd.DataFrame)
    assert list(out_df.index) == ["custom-1", "custom-2"]


def test_endpoint_returns_422_for_invalid_system_payload_shape():
    client = TestClient(app)

    response = client.post(
        "/bui/epc_update_u_values?energy_class=A&archetype=false",
        json={
            "bui": {},
            "system": {
                "heat_emission_data": None,
            },
        },
    )

    assert response.status_code == 422
    detail = response.json().get("detail", "")
    assert "heat_emission_data" in detail
    assert "expected a JSON list or object" in detail
