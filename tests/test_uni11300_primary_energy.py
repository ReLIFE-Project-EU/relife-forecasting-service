from __future__ import annotations

import pandas as pd
import pytest

from relife_forecasting.routes.uni11300_primary_energy import (
    compute_primary_energy_from_hourly_ideal,
)


def test_primary_energy_preserves_rows_with_duplicate_index_labels() -> None:
    hourly = pd.DataFrame(
        {
            "Q_ideal_heat_kWh": [10.0, 20.0, 30.0, 40.0],
            "Q_ideal_cool_kWh": [1.0, 2.0, 3.0, 4.0],
        },
        index=[0, 1, 0, 1],
    )

    result = compute_primary_energy_from_hourly_ideal(hourly)

    assert len(result) == len(hourly)
    assert result.index.tolist() == hourly.index.tolist()
    assert result["Q_ideal_heat_kWh"].sum() == pytest.approx(100.0)
    assert result["Q_ideal_cool_kWh"].sum() == pytest.approx(10.0)
    assert result["EP_total_kWh"].tolist() == pytest.approx(
        (
            result["EP_heat_total_kWh"]
            + result["EP_cool_total_kWh"]
        ).tolist()
    )


def test_primary_energy_preserves_rows_with_unique_index() -> None:
    hourly = pd.DataFrame(
        {
            "Q_ideal_heat_kWh": [5.0, 0.0],
            "Q_ideal_cool_kWh": [0.0, 2.0],
        }
    )

    result = compute_primary_energy_from_hourly_ideal(hourly)

    assert len(result) == 2
    assert result["Q_ideal_heat_kWh"].tolist() == pytest.approx([5.0, 0.0])
    assert result["Q_ideal_cool_kWh"].tolist() == pytest.approx([0.0, 2.0])
