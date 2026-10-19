import json

import numpy as np
import pandas as pd
import pytest

from relife_forecasting.routes.forecasting_service_functions import (
    clean_and_jsonable,
    dataframe_to_records_safe,
    to_jsonable,
)


def _hourly() -> pd.DataFrame:
    index = pd.date_range("2009-01-01", periods=4, freq="h")
    return pd.DataFrame(
        {
            "Q_H": [1.0, 2.0, 3.0, 4.0],
            "T_ve_source_eq": [np.nan] * 4,  # always NaN when H_ve is zero (pybuildingenergy 2.1)
            "X": [1.0, np.inf, -np.inf, 2.0],
        },
        index=index,
    )


@pytest.mark.parametrize("convert", [clean_and_jsonable, to_jsonable])
def test_nan_and_inf_become_null_in_every_converter(convert):
    out = convert(_hourly())
    text = json.dumps(out, allow_nan=False)  # raises on NaN/inf
    assert json.loads(text)["T_ve_source_eq"] == [None] * 4
    assert json.loads(text)["X"] == [1.0, None, None, 2.0]
    assert json.loads(text)["Q_H"] == [1.0, 2.0, 3.0, 4.0]


def test_scalars_and_arrays_with_nan_are_safe():
    payload = {"a": float("nan"), "b": np.float64("inf"), "c": np.array([1.0, np.nan]), "d": [np.nan, 1]}
    json.dumps(clean_and_jsonable(payload), allow_nan=False)
    json.dumps(to_jsonable(payload), allow_nan=False)


def test_records_are_valid_json():
    json.dumps(dataframe_to_records_safe(_hourly()), allow_nan=False)
