from __future__ import annotations

import html
import json
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

try:
    from relife_forecasting.routes.uni11300_primary_energy import (
        CoolingSystemParams,
        HeatingSystemParams,
        compute_primary_energy_from_hourly_ideal,
    )
except Exception:
    from routes.uni11300_primary_energy import (  # type: ignore
        CoolingSystemParams,
        HeatingSystemParams,
        compute_primary_energy_from_hourly_ideal,
    )


def _coerce_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out.sort_index()

    for candidate in ("timestamp", "Unnamed: 0", "datetime", "DateTime", "date"):
        if candidate in out.columns:
            parsed = pd.to_datetime(out[candidate], errors="coerce")
            if parsed.notna().all():
                out = out.drop(columns=[candidate]).set_index(parsed)
                return out.sort_index()

    out.index = pd.date_range("2009-01-01 00:00:00", periods=len(out), freq="h")
    return out.sort_index()


def _monthly_mean(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return pd.DataFrame()
    try:
        return df[cols].resample("ME").mean()
    except Exception:
        return df[cols].resample("M").mean()


def _monthly_sum(df: pd.DataFrame, columns: Iterable[str], scale: float = 1.0) -> pd.DataFrame:
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return pd.DataFrame()
    numeric = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    try:
        monthly = numeric.resample("ME").sum()
    except Exception:
        monthly = numeric.resample("M").sum()
    return monthly * scale


def _month_labels(*dfs: pd.DataFrame) -> List[str]:
    for df in dfs:
        if not df.empty:
            return [idx.strftime("%Y-%m") for idx in df.index]
    return []


def _round_frame(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        try:
            out[col] = pd.to_numeric(out[col])
        except (TypeError, ValueError):
            pass
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def _table_html(df: pd.DataFrame, *, title: str, index_label: str = "Period") -> str:
    if df.empty:
        return f"<section class='table-card'><h3>{html.escape(title)}</h3><p class='empty'>Data not available.</p></section>"

    frame = _round_frame(df)
    headers = [index_label] + list(frame.columns)
    head_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in headers)
    body_rows = []
    for idx, row in frame.iterrows():
        if isinstance(idx, pd.Timestamp):
            idx_text = idx.strftime("%Y-%m")
        else:
            idx_text = str(idx)
        cells = [f"<td>{html.escape(idx_text)}</td>"]
        for value in row.tolist():
            if isinstance(value, float):
                cells.append(f"<td>{value:,.2f}</td>")
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        "<section class='table-card'>"
        f"<h3>{html.escape(title)}</h3>"
        "<div class='table-wrap'>"
        f"<table><thead><tr>{head_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        "</div></section>"
    )


def _safe_percent_saving(baseline: float, scenario: float) -> Optional[float]:
    baseline = float(baseline)
    scenario = float(scenario)
    if abs(baseline) < 1e-9:
        return None
    return ((baseline - scenario) / baseline) * 100.0


def _build_ideal_load_frame(hourly_df: pd.DataFrame) -> pd.DataFrame:
    ideal = pd.DataFrame(index=hourly_df.index)

    if "Q_H" in hourly_df.columns:
        ideal["Q_ideal_heat_kWh"] = pd.to_numeric(hourly_df["Q_H"], errors="coerce").fillna(0.0) * 0.001
    elif "Q_HC" in hourly_df.columns:
        ideal["Q_ideal_heat_kWh"] = pd.to_numeric(hourly_df["Q_HC"], errors="coerce").fillna(0.0).clip(lower=0.0) * 0.001

    if "Q_C" in hourly_df.columns:
        ideal["Q_ideal_cool_kWh"] = pd.to_numeric(hourly_df["Q_C"], errors="coerce").fillna(0.0) * 0.001
    elif "Q_HC" in hourly_df.columns:
        ideal["Q_ideal_cool_kWh"] = (-pd.to_numeric(hourly_df["Q_HC"], errors="coerce").fillna(0.0)).clip(lower=0.0) * 0.001

    return ideal


def _compute_uni_summary(results_df: pd.DataFrame) -> Dict[str, float]:
    summary_fields = [
        "Q_ideal_heat_kWh",
        "Q_ideal_cool_kWh",
        "E_delivered_thermal_kWh",
        "E_delivered_electric_heat_kWh",
        "E_delivered_electric_cool_kWh",
        "E_delivered_electric_total_kWh",
        "EP_heat_total_kWh",
        "EP_cool_total_kWh",
        "EP_total_kWh",
    ]
    return {
        col: float(pd.to_numeric(results_df[col], errors="coerce").fillna(0.0).sum())
        for col in summary_fields
        if col in results_df.columns
    }


def _apply_heat_pump_mask(results_df: pd.DataFrame, generation_mask: Dict[str, Any]) -> pd.DataFrame:
    out = results_df.copy()
    try:
        heat_pump_cop = float(generation_mask.get("heat_pump_cop") or generation_mask.get("mask_value") or 3.2)
    except Exception:
        heat_pump_cop = 3.2
    try:
        fp_electric = float(generation_mask.get("fp_electric") or 2.18)
    except Exception:
        fp_electric = 2.18

    q_ideal_heat = pd.to_numeric(out.get("Q_ideal_heat_kWh", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    e_aux_heat = pd.to_numeric(
        out.get("E_aux_total_heat_kWh", out.get("E_delivered_electric_heat_kWh", 0.0)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    e_electric_cool = pd.to_numeric(out.get("E_delivered_electric_cool_kWh", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    ep_cool = pd.to_numeric(out.get("EP_cool_total_kWh", 0.0), errors="coerce").fillna(0.0)

    e_hp_heating = (q_ideal_heat / heat_pump_cop).clip(lower=0.0)
    e_electric_heat_total = e_hp_heating + e_aux_heat

    out["E_hp_heating_kWh"] = e_hp_heating
    out["E_delivered_electric_heat_kWh"] = e_electric_heat_total
    out["E_delivered_electric_total_kWh"] = e_electric_heat_total + e_electric_cool
    out["EP_heat_total_kWh"] = out["E_delivered_electric_heat_kWh"] * fp_electric
    out["EP_total_kWh"] = out["EP_heat_total_kWh"] + ep_cool
    return out


def _compute_uni_from_building_hourly(
    hourly_df: pd.DataFrame,
    generation_mask: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    if hourly_df.empty:
        return pd.DataFrame(), {}

    generation_mask = generation_mask or {}
    ideal = _build_ideal_load_frame(hourly_df)
    if ideal.empty:
        return pd.DataFrame(), {}

    heating_kwargs: Dict[str, Any] = {}
    if generation_mask.get("applied_mode") != "heat_pump":
        eta_generation = generation_mask.get("heating_eta_generation")
        if eta_generation is None and generation_mask.get("metric") == "eta_generation":
            eta_generation = generation_mask.get("mask_value")
        if eta_generation is not None:
            try:
                heating_kwargs["eta_generation"] = float(eta_generation)
            except Exception:
                pass

    results_df = compute_primary_energy_from_hourly_ideal(
        df_hourly=ideal,
        heat_col="Q_ideal_heat_kWh",
        cool_col="Q_ideal_cool_kWh",
        heating_params=HeatingSystemParams(**heating_kwargs),
        cooling_params=CoolingSystemParams(),
    )

    if generation_mask.get("applied_mode") == "heat_pump":
        results_df = _apply_heat_pump_mask(results_df, generation_mask)

    return results_df, _compute_uni_summary(results_df)


def _annual_summary_frame(
    baseline_iso_annual: Dict[str, float],
    scenario_iso_annual: Dict[str, float],
    baseline_uni_summary: Dict[str, Any],
    scenario_uni_summary: Dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "metric": "Heating need ISO 52016",
            "unit": "kWh",
            "baseline": float(baseline_iso_annual.get("Q_H_kWh", 0.0)),
            "scenario": float(scenario_iso_annual.get("Q_H_kWh", 0.0)),
        },
        {
            "metric": "Cooling need ISO 52016",
            "unit": "kWh",
            "baseline": float(baseline_iso_annual.get("Q_C_kWh", 0.0)),
            "scenario": float(scenario_iso_annual.get("Q_C_kWh", 0.0)),
        },
        {
            "metric": "Primary energy heating UNI/TS 11300",
            "unit": "kWh",
            "baseline": float(baseline_uni_summary.get("EP_heat_total_kWh", 0.0)),
            "scenario": float(scenario_uni_summary.get("EP_heat_total_kWh", 0.0)),
        },
        {
            "metric": "Primary energy total UNI/TS 11300",
            "unit": "kWh",
            "baseline": float(baseline_uni_summary.get("EP_total_kWh", 0.0)),
            "scenario": float(scenario_uni_summary.get("EP_total_kWh", 0.0)),
        },
    ]

    for row in rows:
        row["saving_pct"] = _safe_percent_saving(row["baseline"], row["scenario"])

    return pd.DataFrame(rows)


def _join_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.DataFrame()
    for frame in frames:
        if frame.empty:
            continue
        combined = frame.copy() if combined.empty else combined.join(frame, how="outer")
    return combined


def _build_mask_text(generation_mask: Dict[str, Any]) -> str:
    applied_mode = generation_mask.get("applied_mode") or "default"
    metric = generation_mask.get("metric")
    mask_value = generation_mask.get("mask_value")
    if metric in {None, ""} and mask_value in {None, ""}:
        return str(applied_mode)
    value_text = "-" if mask_value in {None, ""} else str(mask_value)
    return f"{applied_mode} ({metric or '-'}: {value_text})"


def _prepare_report_entry(
    *,
    label: str,
    scenario_meta: Dict[str, Any],
    hourly_df: pd.DataFrame,
    uni_results: Dict[str, Any],
) -> Dict[str, Any]:
    hourly_df = _coerce_datetime_index(hourly_df)
    generation_mask = (
        (uni_results.get("generation_mask") if isinstance(uni_results, dict) else None)
        or scenario_meta.get("generation_mask")
        or {}
    )

    uni_hourly, uni_summary = _compute_uni_from_building_hourly(
        hourly_df,
        generation_mask=generation_mask,
    )
    if uni_hourly.empty:
        uni_hourly = _coerce_datetime_index(pd.DataFrame((uni_results or {}).get("hourly_results") or []))
        uni_summary = (uni_results or {}).get("summary") or {}

    iso_annual = {
        "Q_H_kWh": float(
            pd.to_numeric(hourly_df.get("Q_H", pd.Series(dtype=float)), errors="coerce")
            .fillna(0.0)
            .sum()
            * 0.001
        ),
        "Q_C_kWh": float(
            pd.to_numeric(hourly_df.get("Q_C", pd.Series(dtype=float)), errors="coerce")
            .fillna(0.0)
            .sum()
            * 0.001
        ),
    }

    return {
        "label": label,
        "meta": dict(scenario_meta),
        "generation_mask": generation_mask,
        "hourly": hourly_df,
        "uni_hourly": uni_hourly,
        "uni_summary": uni_summary,
        "iso_annual": iso_annual,
    }


def _annual_summary_multi_frame(
    *,
    baseline_entry: Dict[str, Any],
    scenario_entries: List[Dict[str, Any]],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    metric_specs = [
        ("Heating need ISO 52016", "kWh", lambda entry: float(entry["iso_annual"].get("Q_H_kWh", 0.0))),
        ("Cooling need ISO 52016", "kWh", lambda entry: float(entry["iso_annual"].get("Q_C_kWh", 0.0))),
        ("Primary energy heating UNI/TS 11300", "kWh", lambda entry: float(entry["uni_summary"].get("EP_heat_total_kWh", 0.0))),
        ("Primary energy total UNI/TS 11300", "kWh", lambda entry: float(entry["uni_summary"].get("EP_total_kWh", 0.0))),
    ]

    rows: List[Dict[str, Any]] = []
    chart_payload = {
        "labels": [],
        "baseline": [],
        "scenarios": [
            {"label": entry["label"], "values": [], "savings_pct": []}
            for entry in scenario_entries
        ],
    }

    for metric, unit, getter in metric_specs:
        baseline_value = getter(baseline_entry)
        row: Dict[str, Any] = {"metric": metric, "unit": unit, "Baseline": baseline_value}
        chart_payload["labels"].append(metric)
        chart_payload["baseline"].append(round(baseline_value, 2))

        for idx, entry in enumerate(scenario_entries):
            scenario_value = getter(entry)
            saving = _safe_percent_saving(baseline_value, scenario_value)
            row[entry["label"]] = scenario_value
            row[f"{entry['label']} saving [%]"] = saving
            chart_payload["scenarios"][idx]["values"].append(round(scenario_value, 2))
            chart_payload["scenarios"][idx]["savings_pct"].append(
                None if saving is None else round(float(saving), 2)
            )

        rows.append(row)

    return pd.DataFrame(rows), chart_payload


def build_ecm_multi_scenario_report_html(
    *,
    report_title: str,
    building_meta: Dict[str, Any],
    baseline_hourly: pd.DataFrame,
    baseline_uni_results: Dict[str, Any],
    scenario_contexts: List[Dict[str, Any]],
) -> str:
    baseline_entry = _prepare_report_entry(
        label="Baseline",
        scenario_meta={"label": "Baseline", "elements": [], "generation_mask": baseline_uni_results.get("generation_mask") or {}},
        hourly_df=baseline_hourly,
        uni_results=baseline_uni_results,
    )

    prepared_scenarios: List[Dict[str, Any]] = []
    for idx, context in enumerate(scenario_contexts, start=1):
        scenario_meta = dict(context.get("scenario_meta") or {})
        scenario_label = str(scenario_meta.get("label") or f"Scenario {idx}").strip() or f"Scenario {idx}"
        hourly_payload = context.get("hourly_building")
        prepared_scenarios.append(
            _prepare_report_entry(
                label=scenario_label,
                scenario_meta=scenario_meta,
                hourly_df=hourly_payload if isinstance(hourly_payload, pd.DataFrame) else pd.DataFrame(hourly_payload or []),
                uni_results=context.get("primary_energy_uni11300") or {},
            )
        )

    if not prepared_scenarios:
        raise ValueError("At least one ECM scenario is required to build the report.")

    baseline_temp_monthly = _monthly_mean(baseline_entry["hourly"], ["T_ext", "T_op"]).rename(
        columns={"T_ext": "Baseline T_ext [°C]", "T_op": "Baseline T_op [°C]"}
    )
    temp_monthly = _join_frames(
        [
            baseline_temp_monthly,
            *[
                _monthly_mean(entry["hourly"], ["T_op"]).rename(
                    columns={"T_op": f"{entry['label']} T_op [°C]"}
                )
                for entry in prepared_scenarios
            ],
        ]
    )

    baseline_iso_monthly = _monthly_sum(baseline_entry["hourly"], ["Q_H", "Q_C"], scale=0.001).rename(
        columns={"Q_H": "Baseline Q_H [kWh]", "Q_C": "Baseline Q_C [kWh]"}
    )
    iso_monthly = _join_frames(
        [
            baseline_iso_monthly,
            *[
                _monthly_sum(entry["hourly"], ["Q_H", "Q_C"], scale=0.001).rename(
                    columns={
                        "Q_H": f"{entry['label']} Q_H [kWh]",
                        "Q_C": f"{entry['label']} Q_C [kWh]",
                    }
                )
                for entry in prepared_scenarios
            ],
        ]
    )

    baseline_uni_monthly = _monthly_sum(
        baseline_entry["uni_hourly"],
        ["EP_heat_total_kWh", "EP_cool_total_kWh", "EP_total_kWh"],
    ).rename(
        columns={
            "EP_heat_total_kWh": "Baseline EP_heat [kWh]",
            "EP_cool_total_kWh": "Baseline EP_cool [kWh]",
            "EP_total_kWh": "Baseline EP_total [kWh]",
        }
    )
    uni_monthly = _join_frames(
        [
            baseline_uni_monthly,
            *[
                _monthly_sum(
                    entry["uni_hourly"],
                    ["EP_heat_total_kWh", "EP_cool_total_kWh", "EP_total_kWh"],
                ).rename(
                    columns={
                        "EP_heat_total_kWh": f"{entry['label']} EP_heat [kWh]",
                        "EP_cool_total_kWh": f"{entry['label']} EP_cool [kWh]",
                        "EP_total_kWh": f"{entry['label']} EP_total [kWh]",
                    }
                )
                for entry in prepared_scenarios
            ],
        ]
    )

    annual_summary, annual_chart_payload = _annual_summary_multi_frame(
        baseline_entry=baseline_entry,
        scenario_entries=prepared_scenarios,
    )

    months = _month_labels(temp_monthly, iso_monthly, uni_monthly)
    chart_payload = {
        "hourly_labels": [idx.strftime("%Y-%m-%d %H:%M") for idx in baseline_entry["hourly"].index],
        "temperature": {
            "baseline_t_ext": pd.to_numeric(
                baseline_entry["hourly"].get("T_ext", pd.Series(index=baseline_entry["hourly"].index, dtype=float)),
                errors="coerce",
            ).ffill().fillna(0.0).round(3).tolist(),
            "baseline_t_op": pd.to_numeric(
                baseline_entry["hourly"].get("T_op", pd.Series(index=baseline_entry["hourly"].index, dtype=float)),
                errors="coerce",
            ).ffill().fillna(0.0).round(3).tolist(),
            "scenarios": [
                {
                    "label": entry["label"],
                    "t_op": pd.to_numeric(
                        entry["hourly"].get("T_op", pd.Series(index=entry["hourly"].index, dtype=float)),
                        errors="coerce",
                    ).ffill().fillna(0.0).round(3).tolist(),
                }
                for entry in prepared_scenarios
            ],
        },
        "months": months,
        "iso_monthly": {
            "baseline_qh": iso_monthly.get("Baseline Q_H [kWh]", pd.Series(dtype=float)).fillna(0.0).round(2).tolist(),
            "baseline_qc": iso_monthly.get("Baseline Q_C [kWh]", pd.Series(dtype=float)).fillna(0.0).round(2).tolist(),
            "scenarios": [
                {
                    "label": entry["label"],
                    "qh": iso_monthly.get(f"{entry['label']} Q_H [kWh]", pd.Series(dtype=float)).fillna(0.0).round(2).tolist(),
                    "qc": iso_monthly.get(f"{entry['label']} Q_C [kWh]", pd.Series(dtype=float)).fillna(0.0).round(2).tolist(),
                }
                for entry in prepared_scenarios
            ],
        },
        "uni_monthly": {
            "baseline_ep_total": uni_monthly.get("Baseline EP_total [kWh]", pd.Series(dtype=float)).fillna(0.0).round(2).tolist(),
            "baseline_ep_heat": uni_monthly.get("Baseline EP_heat [kWh]", pd.Series(dtype=float)).fillna(0.0).round(2).tolist(),
            "scenarios": [
                {
                    "label": entry["label"],
                    "ep_total": uni_monthly.get(f"{entry['label']} EP_total [kWh]", pd.Series(dtype=float)).fillna(0.0).round(2).tolist(),
                    "ep_heat": uni_monthly.get(f"{entry['label']} EP_heat [kWh]", pd.Series(dtype=float)).fillna(0.0).round(2).tolist(),
                }
                for entry in prepared_scenarios
            ],
        },
        "annual": annual_chart_payload,
    }

    monthly_tables_html = "".join(
        [
            _table_html(temp_monthly, title="Monthly average temperatures", index_label="Month"),
            _table_html(iso_monthly, title="Monthly ISO 52016 energy needs", index_label="Month"),
            _table_html(uni_monthly, title="Monthly UNI/TS 11300 primary energy", index_label="Month"),
        ]
    )
    annual_table_html = _table_html(
        annual_summary.set_index("metric"),
        title="Annual baseline vs selected scenarios summary" if len(prepared_scenarios) > 1 else "Annual baseline vs ECM summary",
        index_label="Indicator",
    )

    building_title = " / ".join(
        [
            str(building_meta.get("name") or "building"),
            str(building_meta.get("country") or "country"),
            str(building_meta.get("category") or "category"),
        ]
    )
    scenario_labels = [entry["label"] for entry in prepared_scenarios]
    scenario_list_text = ", ".join(scenario_labels)
    scenario_breakdown = " | ".join(
        (
            f"{entry['label']}: "
            f"{', '.join(entry['meta'].get('elements') or []) if entry['meta'].get('elements') else 'generation only'}"
        )
        for entry in prepared_scenarios
    )
    generation_setup = " | ".join(
        f"{entry['label']}: {_build_mask_text(entry['generation_mask'])}"
        for entry in prepared_scenarios
    )

    ep_total_index = annual_chart_payload["labels"].index("Primary energy total UNI/TS 11300")
    best_scenario_text = "n.a."
    best_candidates = [
        (scenario["label"], scenario["savings_pct"][ep_total_index])
        for scenario in annual_chart_payload["scenarios"]
        if scenario["savings_pct"][ep_total_index] is not None
    ]
    if best_candidates:
        best_label, best_value = max(best_candidates, key=lambda item: float(item[1]))
        best_scenario_text = f"{best_label}: {float(best_value):.1f}%"

    cards = [
        ("Compared scenarios" if len(prepared_scenarios) > 1 else "ECM scenario", scenario_list_text),
        ("Scenario count", str(len(prepared_scenarios))),
        ("Applied elements", scenario_breakdown),
        ("Generation setup", generation_setup),
        ("Best total primary energy saving", best_scenario_text),
    ]
    cards_html = "".join(
        "<article class='metric-card'>"
        f"<span class='metric-label'>{html.escape(label)}</span>"
        f"<strong class='metric-value'>{html.escape(str(value))}</strong>"
        "</article>"
        for label, value in cards
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(report_title)}</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{
      --bg: #007F5F;
      --p1: #007F5F;
      --p2: #2B9348;
      --p3: #55A630;
      --p4: #80B918;
      --p5: #AACC00;
      --p6: #BFD200;
      --p7: #D4D700;
      --p8: #DDDF00;
      --p9: #EEEF20;
      --p10: #FFFF3F;
      --panel: #ffffff;
      --panel-strong: rgba(255, 255, 255, 0.92);
      --ink: #007F5F;
      --muted: #2B9348;
      --accent: #ffffff;
      --accent-2: #ffffff;
      --accent-3: #ffffff;
      --line: rgba(0, 127, 95, 0.16);
      --shadow: 0 18px 40px rgba(0, 127, 95, 0.16);
      --font-brand: "Indivisible", Arial, sans-serif;
    }}
    @font-face {{
      font-family: "Indivisible";
      src: local("Indivisible"), local("Indivisible Regular");
      font-style: normal;
      font-weight: 400;
      font-display: swap;
    }}
    @font-face {{
      font-family: "Indivisible";
      src: local("Indivisible Bold"), local("Indivisible");
      font-style: normal;
      font-weight: 700;
      font-display: swap;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--font-brand);
      background: var(--p1);
      background: linear-gradient(
        180deg,
        var(--p1) 0%,
        var(--p2) 12%,
        var(--p3) 24%,
        var(--p4) 36%,
        var(--p5) 52%,
        var(--p6) 64%,
        var(--p7) 76%,
        var(--p8) 86%,
        var(--p9) 94%,
        var(--p10) 100%
      );
      color: var(--ink);
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid rgba(255, 255, 255, 0.55);
      border-radius: 28px;
      padding: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(4px);
    }}
    .eyebrow {{
      margin: 0 0 8px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 12px;
      color: var(--p1);
      font-weight: 700;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(30px, 4vw, 54px);
      line-height: 1.02;
      font-family: var(--font-brand);
      font-weight: 700;
    }}
    .subtitle {{
      margin: 14px 0 0;
      max-width: 920px;
      font-size: 18px;
      line-height: 1.6;
      color: var(--muted);
    }}
    .meta-grid, .metric-grid {{
      display: grid;
      gap: 16px;
      margin-top: 24px;
    }}
    .meta-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .metric-grid {{
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }}
    .meta-card, .metric-card, .chart-card, .table-card {{
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.6);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .meta-card, .metric-card {{
      padding: 18px 20px;
    }}
    .meta-label, .metric-label {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .meta-value, .metric-value {{
      font-size: 20px;
      line-height: 1.35;
      color: var(--ink);
      font-family: var(--font-brand);
    }}
    .section-title {{
      margin: 34px 0 18px;
      font-size: 28px;
      font-family: var(--font-brand);
      font-weight: 700;
    }}
    .chart-grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: 1fr;
    }}
    .chart-card {{
      padding: 20px 20px 12px;
    }}
    .chart-card h3, .table-card h3 {{
      margin: 0 0 12px;
      font-size: 20px;
      font-family: var(--font-brand);
      font-weight: 700;
    }}
    .chart-note {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .chart {{
      width: 100%;
      min-height: 420px;
    }}
    .table-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .table-card {{
      padding: 18px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }}
    thead {{
      background: rgba(170, 204, 0, 0.18);
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
    }}
    th {{
      font-family: var(--font-brand);
      font-weight: 700;
    }}
    tbody tr:nth-child(even) {{
      background: rgba(212, 215, 0, 0.1);
    }}
    .empty {{
      color: var(--muted);
      margin: 0;
    }}
    @media (max-width: 720px) {{
      .page {{ padding: 16px; }}
      .hero {{ padding: 22px; border-radius: 20px; }}
      .chart {{ min-height: 320px; }}
      table {{ min-width: 640px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <p class="eyebrow">ECM Comparison Report</p>
      <h1>{html.escape(report_title)}</h1>
      <p class="subtitle">
        Comparison between the baseline and the selected renovation scenarios for <strong>{html.escape(building_title)}</strong>.
        Included scenarios: <strong>{html.escape(scenario_list_text)}</strong>. The report shows hourly temperatures,
        ISO 52016 energy needs, UNI/TS 11300 primary energy and savings against the baseline in the same charts.
      </p>
      <div class="meta-grid">
        <article class="meta-card"><span class="meta-label">Building</span><strong class="meta-value">{html.escape(str(building_meta.get("name") or "-"))}</strong></article>
        <article class="meta-card"><span class="meta-label">Category</span><strong class="meta-value">{html.escape(str(building_meta.get("category") or "-"))}</strong></article>
        <article class="meta-card"><span class="meta-label">Country</span><strong class="meta-value">{html.escape(str(building_meta.get("country") or "-"))}</strong></article>
        <article class="meta-card"><span class="meta-label">Weather Source</span><strong class="meta-value">{html.escape(str(building_meta.get("weather_source") or "-"))}</strong></article>
      </div>
      <div class="metric-grid">{cards_html}</div>
    </section>

    <h2 class="section-title">Charts</h2>
    <section class="chart-grid">
      <article class="chart-card">
        <h3>Hourly temperature trends</h3>
        <p class="chart-note">Baseline operative temperature and all selected scenario operative temperatures with the outdoor reference temperature.</p>
        <div id="temperatureChart" class="chart"></div>
      </article>
      <article class="chart-card">
        <h3>Monthly ISO 52016 energy needs</h3>
        <p class="chart-note">Monthly grouped comparison between baseline and all selected scenarios for heating and cooling.</p>
        <div id="isoNeedChart" class="chart"></div>
      </article>
      <article class="chart-card">
        <h3>Monthly UNI/TS 11300 primary energy</h3>
        <p class="chart-note">Monthly grouped comparison of total and heating primary energy for the baseline and selected scenarios.</p>
        <div id="uniChart" class="chart"></div>
      </article>
      <article class="chart-card">
        <h3>Annual comparison and savings</h3>
        <p class="chart-note">Grouped bars show the baseline and each selected scenario; dashed lines show savings against the baseline.</p>
        <div id="annualChart" class="chart"></div>
      </article>
    </section>

    <h2 class="section-title">Tables</h2>
    <section class="table-grid">
      {monthly_tables_html}
      {annual_table_html}
    </section>
  </main>

  <script>
    const reportData = {json.dumps(chart_payload)};
    const palette = {{
      p1: '#007F5F',
      p2: '#2B9348',
      p3: '#55A630',
      p4: '#80B918',
      p5: '#AACC00',
      p6: '#BFD200',
      p7: '#D4D700',
      p8: '#DDDF00',
      p9: '#EEEF20',
      p10: '#FFFF3F'
    }};
    const scenarioBarColors = [palette.p3, palette.p4, palette.p5, palette.p6, palette.p7, palette.p8, palette.p9, palette.p10];
    const scenarioLineColors = [palette.p4, palette.p5, palette.p6, palette.p7, palette.p8, palette.p9, palette.p10, palette.p2];

    function createChart(id, option) {{
      const chart = echarts.init(document.getElementById(id));
      chart.setOption(option);
      window.addEventListener('resize', () => chart.resize());
      return chart;
    }}

    function barColor(idx) {{
      return scenarioBarColors[idx % scenarioBarColors.length];
    }}

    function lineColor(idx) {{
      return scenarioLineColors[idx % scenarioLineColors.length];
    }}

    const temperatureSeries = [
      {{
        name: 'Baseline outdoor temperature',
        type: 'line',
        showSymbol: false,
        smooth: true,
        lineStyle: {{ width: 1.5, color: palette.p6 }},
        itemStyle: {{ color: palette.p6 }},
        data: reportData.temperature.baseline_t_ext
      }},
      {{
        name: 'Baseline operative temperature',
        type: 'line',
        showSymbol: false,
        smooth: true,
        lineStyle: {{ width: 2, color: palette.p1 }},
        itemStyle: {{ color: palette.p1 }},
        data: reportData.temperature.baseline_t_op
      }},
      ...reportData.temperature.scenarios.map((scenario, idx) => ({{
        name: `${{scenario.label}} operative temperature`,
        type: 'line',
        showSymbol: false,
        smooth: true,
        lineStyle: {{ width: 2, color: lineColor(idx) }},
        itemStyle: {{ color: lineColor(idx) }},
        data: scenario.t_op
      }}))
    ];

    createChart('temperatureChart', {{
      tooltip: {{ trigger: 'axis' }},
      legend: {{ top: 0 }},
      grid: {{ left: 56, right: 26, top: 48, bottom: 72 }},
      xAxis: {{
        type: 'category',
        data: reportData.hourly_labels,
        boundaryGap: false,
      }},
      yAxis: {{
        type: 'value',
        name: '°C',
      }},
      dataZoom: [
        {{ type: 'inside', start: 0, end: 12 }},
        {{ start: 0, end: 12, bottom: 16 }}
      ],
      series: temperatureSeries
    }});

    const isoSeries = [
      {{
        name: 'Baseline Q_H',
        type: 'bar',
        itemStyle: {{ color: palette.p1, borderColor: palette.p2, borderWidth: 1 }},
        data: reportData.iso_monthly.baseline_qh
      }},
      {{
        name: 'Baseline Q_C',
        type: 'line',
        smooth: true,
        lineStyle: {{ width: 2, color: palette.p2 }},
        itemStyle: {{ color: palette.p2 }},
        data: reportData.iso_monthly.baseline_qc
      }},
      ...reportData.iso_monthly.scenarios.flatMap((scenario, idx) => ([
        {{
          name: `${{scenario.label}} Q_H`,
          type: 'bar',
          itemStyle: {{ color: barColor(idx), borderColor: palette.p1, borderWidth: 0.6 }},
          data: scenario.qh
        }},
        {{
          name: `${{scenario.label}} Q_C`,
          type: 'line',
          smooth: true,
          lineStyle: {{ width: 2, color: lineColor(idx) }},
          itemStyle: {{ color: lineColor(idx) }},
          data: scenario.qc
        }}
      ]))
    ];

    createChart('isoNeedChart', {{
      tooltip: {{ trigger: 'axis' }},
      legend: {{ top: 0 }},
      grid: {{ left: 56, right: 24, top: 48, bottom: 46 }},
      xAxis: {{ type: 'category', data: reportData.months }},
      yAxis: {{ type: 'value', name: 'kWh' }},
      series: isoSeries
    }});

    const uniSeries = [
      {{
        name: 'Baseline total EP',
        type: 'bar',
        itemStyle: {{ color: palette.p1, borderColor: palette.p2, borderWidth: 1 }},
        data: reportData.uni_monthly.baseline_ep_total
      }},
      {{
        name: 'Baseline heating EP',
        type: 'line',
        smooth: true,
        lineStyle: {{ width: 2, color: palette.p2 }},
        itemStyle: {{ color: palette.p2 }},
        data: reportData.uni_monthly.baseline_ep_heat
      }},
      ...reportData.uni_monthly.scenarios.flatMap((scenario, idx) => ([
        {{
          name: `${{scenario.label}} total EP`,
          type: 'bar',
          itemStyle: {{ color: barColor(idx), borderColor: palette.p1, borderWidth: 0.6 }},
          data: scenario.ep_total
        }},
        {{
          name: `${{scenario.label}} heating EP`,
          type: 'line',
          smooth: true,
          lineStyle: {{ width: 2, color: lineColor(idx) }},
          itemStyle: {{ color: lineColor(idx) }},
          data: scenario.ep_heat
        }}
      ]))
    ];

    createChart('uniChart', {{
      tooltip: {{ trigger: 'axis' }},
      legend: {{ top: 0 }},
      grid: {{ left: 56, right: 24, top: 48, bottom: 46 }},
      xAxis: {{ type: 'category', data: reportData.months }},
      yAxis: {{ type: 'value', name: 'kWh' }},
      series: uniSeries
    }});

    const annualSeries = [
      {{
        name: 'Baseline',
        type: 'bar',
        itemStyle: {{ color: palette.p1, borderColor: palette.p2, borderWidth: 1 }},
        data: reportData.annual.baseline
      }},
      ...reportData.annual.scenarios.map((scenario, idx) => ({{
        name: scenario.label,
        type: 'bar',
        itemStyle: {{ color: barColor(idx) }},
        data: scenario.values
      }})),
      ...reportData.annual.scenarios.map((scenario, idx) => ({{
        name: `${{scenario.label}} saving %`,
        type: 'line',
        yAxisIndex: 1,
        smooth: true,
        lineStyle: {{ width: 2, type: 'dashed', color: lineColor(idx) }},
        itemStyle: {{ color: lineColor(idx) }},
        label: {{
          show: reportData.annual.scenarios.length === 1,
          formatter: (params) => params.value == null ? 'n.a.' : `${{params.value}}%`
        }},
        data: scenario.savings_pct
      }}))
    ];

    createChart('annualChart', {{
      tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
      legend: {{ top: 0 }},
      grid: {{ left: 56, right: 56, top: 52, bottom: 56 }},
      xAxis: {{ type: 'category', data: reportData.annual.labels }},
      yAxis: [
        {{ type: 'value', name: 'kWh' }},
        {{ type: 'value', name: 'Saving %', axisLabel: {{ formatter: '{{value}}%' }} }}
      ],
      series: annualSeries
    }});
  </script>
</body>
</html>"""


def build_ecm_comparison_report_html(
    *,
    report_title: str,
    building_meta: Dict[str, Any],
    scenario_meta: Dict[str, Any],
    baseline_hourly: pd.DataFrame,
    scenario_hourly: pd.DataFrame,
    baseline_uni_results: Dict[str, Any],
    scenario_uni_results: Dict[str, Any],
) -> str:
    return build_ecm_multi_scenario_report_html(
        report_title=report_title,
        building_meta=building_meta,
        baseline_hourly=baseline_hourly,
        baseline_uni_results=baseline_uni_results,
        scenario_contexts=[
            {
                "scenario_meta": scenario_meta,
                "hourly_building": scenario_hourly,
                "primary_energy_uni11300": scenario_uni_results,
            }
        ],
    )
