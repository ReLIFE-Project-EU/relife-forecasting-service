# serializers.py
from typing import Any
import math

import numpy as np
import pandas as pd


def _sanitize_scalar(x: Any) -> Any:
    """
    Rende JSON-safe gli scalari numerici:
    - float/np.floating con NaN/inf -> None
    - float normale -> float(x)
    - np.integer -> int(x)
    Altri tipi li lascia invariati.
    """
    # Numeri floating (Python o NumPy)
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        if not math.isfinite(xf):  # NaN, +inf, -inf
            return None
        return xf

    # Interi NumPy -> int
    if isinstance(x, np.integer):
        return int(x)

    return x


def to_jsonable(obj: Any) -> Any:
    """
    Converte ricorsivamente obj in una struttura composta solo da tipi JSON-serializzabili.
    Gestisce anche NaN/inf -> None.
    - np.ndarray -> list
    - pd.DataFrame -> list[dict]
    - pd.Series -> list
    - dict, list, tuple -> ricorsione
    - float/np.floating con NaN/inf -> None
    """
    # np.ndarray
    if isinstance(obj, np.ndarray):
        # prima lo trasformo in lista, poi ricorsione sugli elementi
        return [to_jsonable(v) for v in obj.tolist()]

    # DataFrame
    if isinstance(obj, pd.DataFrame):
        records = obj.to_dict(orient="records")
        return [to_jsonable(row) for row in records]

    # Series
    if isinstance(obj, pd.Series):
        return [to_jsonable(v) for v in obj.to_list()]

    # dict
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    # scalari numerici (float/np.floating/np.integer)
    obj = _sanitize_scalar(obj)

    # Tutto il resto (str, bool, None, ecc.)
    return obj
