from __future__ import annotations

import pandas as pd
import pytest

from relife_forecasting.routes.forecasting_service_functions import (
    drop_iso52016_warmup_rows,
)


def _one_year() -> pd.DataFrame:
    """An hourly year whose Q_H values identify each row by position."""
    index = pd.date_range("2009-01-01 00:00:00", periods=8760, freq="h")
    return pd.DataFrame({"Q_H": range(8760)}, index=index)


def _december(year: pd.DataFrame) -> pd.DataFrame:
    return year[year.index.month == 12]


def test_drops_the_prepended_warmup_month() -> None:
    year = _one_year()
    with_warmup = pd.concat([_december(year), year])

    assert len(with_warmup) == 9504

    result = drop_iso52016_warmup_rows(with_warmup)

    assert len(result) == 8760
    assert result.index.is_unique
    assert result.index[0] == pd.Timestamp("2009-01-01 00:00:00")
    assert result.index[-1] == pd.Timestamp("2009-12-31 23:00:00")
    # The copies were dropped, not the real December.
    assert result["Q_H"].tolist() == year["Q_H"].tolist()


def test_is_a_no_op_when_there_is_no_warmup() -> None:
    """Guards against eating real hours once pybuildingenergy trims warm-up itself."""
    year = _one_year()

    result = drop_iso52016_warmup_rows(year)

    assert len(result) == 8760
    assert result["Q_H"].tolist() == year["Q_H"].tolist()


def test_rejects_duplicates_that_are_not_a_leading_block() -> None:
    year = _one_year()
    scattered = pd.concat([year, year.iloc[5000:5010]]).sort_index()

    with pytest.raises(ValueError, match="not a leading block"):
        drop_iso52016_warmup_rows(scattered)


def test_rejects_a_warmup_period_the_copy_check_cannot_see() -> None:
    """A warm-up whose labels do not collide is caught by the row count."""
    year = _one_year()
    relabelled = _december(year).copy()
    relabelled.index = relabelled.index.map(lambda ts: ts.replace(year=2008))

    with pytest.raises(ValueError, match="expected 8760 or 8784"):
        drop_iso52016_warmup_rows(pd.concat([relabelled, year]))
