from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union

import pandas as pd

NumberOrSeries = Union[float, pd.Series]


def _as_series(x: NumberOrSeries, index: pd.Index) -> pd.Series:
    """Convert a scalar or Series to a Series aligned with the given index."""
    if isinstance(x, pd.Series):
        s = x.reindex(index)
        if s.isna().any():
            raise ValueError("Series has NaN after reindex; check time alignment.")
        return s
    return pd.Series(float(x), index=index)


def _clip_nonnegative(s: pd.Series, name: str) -> pd.Series:
    """Ensure non-negative series (ideal loads must be >= 0)."""
    if (s < 0).any():
        raise ValueError(f"{name} contains negative values; expected >= 0.")
    return s


def _validate_eta(eta: pd.Series, name: str, allow_gt1: bool = False) -> None:
    """Validate efficiency ranges."""
    if (eta <= 0).any():
        raise ValueError(f"{name} must be > 0.")
    if not allow_gt1 and (eta > 1).any():
        raise ValueError(f"{name} must be <= 1 (found values > 1).")


def _validate_fraction(f: pd.Series, name: str) -> None:
    """Validate fraction in [0, 1]."""
    if (f < 0).any() or (f > 1).any():
        raise ValueError(f"{name} must be in [0, 1].")


@dataclass(frozen=True)
class HeatingSystemParams:
    """
    Heating parameters (UNI/TS 11300-2).

    Efficiencies (eta):
      - eta_emission
      - eta_distribution
      - eta_storage
      - eta_generation

    Recoverable loss fractions:
      - f_recov_emission
      - f_recov_distribution
      - f_recov_storage

    Auxiliary electric fractions:
      - aux_emission_fraction
      - aux_distribution_fraction
      - aux_storage_fraction
      - aux_generation_fraction

    Primary energy factors:
      - fp_thermal
      - fp_electric
    """
    eta_emission: NumberOrSeries = 0.95
    eta_distribution: NumberOrSeries = 0.93
    eta_storage: NumberOrSeries = 0.98
    eta_generation: NumberOrSeries = 0.92

    f_recov_emission: NumberOrSeries = 1.00
    f_recov_distribution: NumberOrSeries = 0.80
    f_recov_storage: NumberOrSeries = 0.90

    aux_emission_fraction: NumberOrSeries = 0.005
    aux_distribution_fraction: NumberOrSeries = 0.010
    aux_storage_fraction: NumberOrSeries = 0.001
    aux_generation_fraction: NumberOrSeries = 0.015

    fp_thermal: NumberOrSeries = 1.05
    fp_electric: NumberOrSeries = 2.18


@dataclass(frozen=True)
class CoolingSystemParams:
    """
    Cooling parameters (UNI/TS 11300-3).

    Efficiencies and COP:
      - eta_emission
      - eta_distribution
      - eta_storage
      - cop_generation

    Recoverable loss fractions:
      - f_recov_emission
      - f_recov_distribution
      - f_recov_storage

    Auxiliary electric fractions:
      - aux_emission_fraction
      - aux_distribution_fraction
      - aux_storage_fraction
      - aux_generation_fraction

    Primary energy factor:
      - fp_electric
    """
    eta_emission: NumberOrSeries = 0.97
    eta_distribution: NumberOrSeries = 0.95
    eta_storage: NumberOrSeries = 1.00
    cop_generation: NumberOrSeries = 3.2

    f_recov_emission: NumberOrSeries = 1.00
    f_recov_distribution: NumberOrSeries = 0.80
    f_recov_storage: NumberOrSeries = 0.90

    aux_emission_fraction: NumberOrSeries = 0.005
    aux_distribution_fraction: NumberOrSeries = 0.010
    aux_storage_fraction: NumberOrSeries = 0.001
    aux_generation_fraction: NumberOrSeries = 0.005

    fp_electric: NumberOrSeries = 2.18


def compute_heating_from_ideal(
    q_ideal_heat_kwh: pd.Series,
    params: HeatingSystemParams,
) -> pd.DataFrame:
    """
    Compute delivered and primary energy for heating from hourly ideal loads.
    """
    idx = q_ideal_heat_kwh.index
    q_ideal = _clip_nonnegative(q_ideal_heat_kwh.astype(float), "Q_ideal_heat_kWh")

    eta_em = _as_series(params.eta_emission, idx)
    eta_dist = _as_series(params.eta_distribution, idx)
    eta_acc = _as_series(params.eta_storage, idx)
    eta_gen = _as_series(params.eta_generation, idx)

    f_recov_em = _as_series(params.f_recov_emission, idx)
    f_recov_dist = _as_series(params.f_recov_distribution, idx)
    f_recov_acc = _as_series(params.f_recov_storage, idx)

    aux_em = _as_series(params.aux_emission_fraction, idx)
    aux_dist = _as_series(params.aux_distribution_fraction, idx)
    aux_acc = _as_series(params.aux_storage_fraction, idx)
    aux_gen = _as_series(params.aux_generation_fraction, idx)

    fp_th = _as_series(params.fp_thermal, idx)
    fp_el = _as_series(params.fp_electric, idx)

    _validate_eta(eta_em, "eta_emission")
    _validate_eta(eta_dist, "eta_distribution")
    _validate_eta(eta_acc, "eta_storage")
    _validate_eta(eta_gen, "eta_generation")

    _validate_fraction(f_recov_em, "f_recov_emission")
    _validate_fraction(f_recov_dist, "f_recov_distribution")
    _validate_fraction(f_recov_acc, "f_recov_storage")

    _validate_fraction(aux_em, "aux_emission_fraction")
    _validate_fraction(aux_dist, "aux_distribution_fraction")
    _validate_fraction(aux_acc, "aux_storage_fraction")
    _validate_fraction(aux_gen, "aux_generation_fraction")

    _validate_eta(fp_th, "fp_thermal", allow_gt1=True)
    _validate_eta(fp_el, "fp_electric", allow_gt1=True)

    Q_em_out = q_ideal / eta_em
    Q_loss_em = Q_em_out - q_ideal
    Q_loss_em_recov = Q_loss_em * f_recov_em

    Q_dist_net = Q_em_out - Q_loss_em_recov
    Q_dist_out = Q_dist_net / eta_dist
    Q_loss_dist = Q_dist_out - Q_dist_net
    Q_loss_dist_recov = Q_loss_dist * f_recov_dist

    Q_acc_net = Q_dist_out - Q_loss_dist_recov
    Q_acc_out = Q_acc_net / eta_acc
    Q_loss_acc = Q_acc_out - Q_acc_net
    Q_loss_acc_recov = Q_loss_acc * f_recov_acc

    Q_gen_net = Q_acc_out - Q_loss_acc_recov
    Q_gen_in = Q_gen_net / eta_gen

    E_aux_em = q_ideal * aux_em
    E_aux_dist = Q_em_out * aux_dist
    E_aux_acc = Q_dist_out * aux_acc
    E_aux_gen = Q_gen_in * aux_gen
    E_aux_total = E_aux_em + E_aux_dist + E_aux_acc + E_aux_gen

    E_delivered_thermal = Q_gen_in
    E_delivered_electric = E_aux_total

    EP_thermal = E_delivered_thermal * fp_th
    EP_electric = E_delivered_electric * fp_el
    EP_total = EP_thermal + EP_electric

    return pd.DataFrame(
        {
            "Q_ideal_heat_kWh": q_ideal,
            "Q_emission_out_heat_kWh": Q_em_out,
            "Q_loss_emission_heat_kWh": Q_loss_em,
            "Q_loss_emission_recov_heat_kWh": Q_loss_em_recov,
            "Q_distribution_out_heat_kWh": Q_dist_out,
            "Q_loss_distribution_heat_kWh": Q_loss_dist,
            "Q_loss_distribution_recov_heat_kWh": Q_loss_dist_recov,
            "Q_storage_out_heat_kWh": Q_acc_out,
            "Q_loss_storage_heat_kWh": Q_loss_acc,
            "Q_loss_storage_recov_heat_kWh": Q_loss_acc_recov,
            "Q_generation_in_heat_kWh": Q_gen_in,
            "E_aux_emission_heat_kWh": E_aux_em,
            "E_aux_distribution_heat_kWh": E_aux_dist,
            "E_aux_storage_heat_kWh": E_aux_acc,
            "E_aux_generation_heat_kWh": E_aux_gen,
            "E_aux_total_heat_kWh": E_aux_total,
            "E_delivered_thermal_kWh": E_delivered_thermal,
            "E_delivered_electric_heat_kWh": E_delivered_electric,
            "EP_thermal_kWh": EP_thermal,
            "EP_electric_heat_kWh": EP_electric,
            "EP_heat_total_kWh": EP_total,
        },
        index=idx,
    )


def compute_cooling_from_ideal(
    q_ideal_cool_kwh: pd.Series,
    params: CoolingSystemParams,
) -> pd.DataFrame:
    """
    Compute delivered and primary energy for cooling from hourly ideal loads.
    """
    idx = q_ideal_cool_kwh.index
    q_ideal = _clip_nonnegative(q_ideal_cool_kwh.astype(float), "Q_ideal_cool_kWh")

    eta_em = _as_series(params.eta_emission, idx)
    eta_dist = _as_series(params.eta_distribution, idx)
    eta_acc = _as_series(params.eta_storage, idx)
    cop = _as_series(params.cop_generation, idx)

    f_recov_em = _as_series(params.f_recov_emission, idx)
    f_recov_dist = _as_series(params.f_recov_distribution, idx)
    f_recov_acc = _as_series(params.f_recov_storage, idx)

    aux_em = _as_series(params.aux_emission_fraction, idx)
    aux_dist = _as_series(params.aux_distribution_fraction, idx)
    aux_acc = _as_series(params.aux_storage_fraction, idx)
    aux_gen = _as_series(params.aux_generation_fraction, idx)

    fp_el = _as_series(params.fp_electric, idx)

    _validate_eta(eta_em, "eta_emission")
    _validate_eta(eta_dist, "eta_distribution")
    _validate_eta(eta_acc, "eta_storage")
    _validate_eta(cop, "cop_generation", allow_gt1=True)

    _validate_fraction(f_recov_em, "f_recov_emission")
    _validate_fraction(f_recov_dist, "f_recov_distribution")
    _validate_fraction(f_recov_acc, "f_recov_storage")

    _validate_fraction(aux_em, "aux_emission_fraction")
    _validate_fraction(aux_dist, "aux_distribution_fraction")
    _validate_fraction(aux_acc, "aux_storage_fraction")
    _validate_fraction(aux_gen, "aux_generation_fraction")

    _validate_eta(fp_el, "fp_electric", allow_gt1=True)

    Q_em_out = q_ideal / eta_em
    Q_loss_em = Q_em_out - q_ideal
    Q_loss_em_recov = Q_loss_em * f_recov_em

    Q_dist_net = Q_em_out + Q_loss_em_recov
    Q_dist_out = Q_dist_net / eta_dist
    Q_loss_dist = Q_dist_out - Q_dist_net
    Q_loss_dist_recov = Q_loss_dist * f_recov_dist

    Q_acc_net = Q_dist_out + Q_loss_dist_recov
    Q_acc_out = Q_acc_net / eta_acc
    Q_loss_acc = Q_acc_out - Q_acc_net
    Q_loss_acc_recov = Q_loss_acc * f_recov_acc

    Q_gen_net = Q_acc_out + Q_loss_acc_recov
    E_el_gen = Q_gen_net / cop

    E_aux_em = q_ideal * aux_em
    E_aux_dist = Q_em_out * aux_dist
    E_aux_acc = Q_dist_out * aux_acc
    E_aux_gen = E_el_gen * aux_gen
    E_aux_total = E_aux_em + E_aux_dist + E_aux_acc + E_aux_gen

    E_delivered_electric = E_el_gen + E_aux_total
    EP_electric = E_delivered_electric * fp_el

    return pd.DataFrame(
        {
            "Q_ideal_cool_kWh": q_ideal,
            "Q_emission_out_cool_kWh": Q_em_out,
            "Q_loss_emission_cool_kWh": Q_loss_em,
            "Q_loss_emission_recov_cool_kWh": Q_loss_em_recov,
            "Q_distribution_out_cool_kWh": Q_dist_out,
            "Q_loss_distribution_cool_kWh": Q_loss_dist,
            "Q_loss_distribution_recov_cool_kWh": Q_loss_dist_recov,
            "Q_storage_out_cool_kWh": Q_acc_out,
            "Q_loss_storage_cool_kWh": Q_loss_acc,
            "Q_loss_storage_recov_cool_kWh": Q_loss_acc_recov,
            "Q_generation_net_cool_kWh": Q_gen_net,
            "E_generation_electric_cool_kWh": E_el_gen,
            "E_aux_emission_cool_kWh": E_aux_em,
            "E_aux_distribution_cool_kWh": E_aux_dist,
            "E_aux_storage_cool_kWh": E_aux_acc,
            "E_aux_generation_cool_kWh": E_aux_gen,
            "E_aux_total_cool_kWh": E_aux_total,
            "E_delivered_electric_cool_kWh": E_delivered_electric,
            "EP_electric_cool_kWh": EP_electric,
            "EP_cool_total_kWh": EP_electric,
        },
        index=idx,
    )


def compute_primary_energy_from_hourly_ideal(
    df_hourly: pd.DataFrame,
    heat_col: str = "Q_ideal_heat_kWh",
    cool_col: str = "Q_ideal_cool_kWh",
    heating_params: Optional[HeatingSystemParams] = None,
    cooling_params: Optional[CoolingSystemParams] = None,
) -> pd.DataFrame:
    """
    Wrapper to compute delivered and primary energy from ideal loads.
    """
    if heating_params is None:
        heating_params = HeatingSystemParams()
    if cooling_params is None:
        cooling_params = CoolingSystemParams()

    out = pd.DataFrame(index=df_hourly.index)

    if heat_col in df_hourly.columns:
        heat_results = compute_heating_from_ideal(df_hourly[heat_col], heating_params)
        out = out.join(heat_results, how="left")

    if cool_col in df_hourly.columns:
        cool_results = compute_cooling_from_ideal(df_hourly[cool_col], cooling_params)
        out = out.join(cool_results, how="left")

    if "EP_heat_total_kWh" in out.columns and "EP_cool_total_kWh" in out.columns:
        out["EP_total_kWh"] = out["EP_heat_total_kWh"].fillna(0.0) + out["EP_cool_total_kWh"].fillna(0.0)

    if "E_delivered_electric_heat_kWh" in out.columns and "E_delivered_electric_cool_kWh" in out.columns:
        out["E_delivered_electric_total_kWh"] = (
            out["E_delivered_electric_heat_kWh"].fillna(0.0)
            + out["E_delivered_electric_cool_kWh"].fillna(0.0)
        )

    return out


def build_uni11300_input_example() -> Dict[str, Any]:
    """Return a compact input example for the UNI/TS 11300 calculation."""
    return {
        "hourly_sim": [
            {"timestamp": "2024-01-01 00:00:00", "Q_H": 2103.85, "Q_C": 0.0},
            {"timestamp": "2024-01-01 01:00:00", "Q_H": 1800.12, "Q_C": 0.0},
        ],
        "input_unit": "Wh",
        "heating_params": asdict(HeatingSystemParams()),
        "cooling_params": asdict(CoolingSystemParams()),
    }
