#!/usr/bin/env python3
"""
Script unico per:
- leggere i risultati delle simulazioni (building + system heating)
- calcolare consumi annuali del sistema
- calcolare emissioni CO2eq e risparmi tra scenari
- generare tabelle (CSV/HTML) e grafici di confronto con pyecharts
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from pyecharts.globals import ThemeType
from pyecharts.charts import Page, Bar, Line
from pyecharts import options as opts

# Se usi funzioni custom per i bar chart singoli
from utils.graphs import bar_chart_single
from building_examples import BUI


# ==============================================================================
#                      SEZIONE CO₂ – MODELS & LOGIC
# ==============================================================================

class EnergySource(str, Enum):
    """Fonti energetiche disponibili"""
    GRID_ELECTRICITY = "grid_electricity"
    NATURAL_GAS = "natural_gas"
    LPG = "lpg"
    DIESEL = "diesel"
    BIOMASS = "biomass"
    DISTRICT_HEATING = "district_heating"
    SOLAR_PV = "solar_pv"
    WIND = "wind"
    HEAT_PUMP_ELECTRIC = "heat_pump_electric"


# Fattori di emissione in kgCO2eq/kWh
EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "IT": {  # Italia
        "grid_electricity": 0.280,
        "natural_gas": 0.202,
        "lpg": 0.234,
        "diesel": 0.267,
        "biomass": 0.030,
        "district_heating": 0.180,
        "solar_pv": 0.040,
        "wind": 0.012,
        "heat_pump_electric": 0.070,
    },
    "EU": {  # Media europea
        "grid_electricity": 0.255,
        "natural_gas": 0.202,
        "lpg": 0.234,
        "diesel": 0.267,
        "biomass": 0.030,
        "district_heating": 0.150,
        "solar_pv": 0.040,
        "wind": 0.012,
        "heat_pump_electric": 0.064,
    },
    "DE": {  # Germania
        "grid_electricity": 0.420,
        "natural_gas": 0.202,
        "lpg": 0.234,
        "diesel": 0.267,
        "biomass": 0.030,
        "district_heating": 0.200,
        "solar_pv": 0.040,
        "wind": 0.012,
        "heat_pump_electric": 0.105,
    },
}


def calculate_emissions(
    energy_source: EnergySource | str,
    annual_consumption_kwh: float,
    country: str = "IT",
) -> Dict[str, float]:
    """
    Calcola emissioni CO2eq e parametri equivalenti.
    """
    if country not in EMISSION_FACTORS:
        country = "IT"

    if isinstance(energy_source, EnergySource):
        source_key = energy_source.value
    else:
        source_key = str(energy_source)

    emission_factor = EMISSION_FACTORS[country].get(
        source_key,
        EMISSION_FACTORS[country]["grid_electricity"],
    )

    annual_emissions_kg = annual_consumption_kwh * emission_factor
    annual_emissions_ton = annual_emissions_kg / 1000.0

    equivalent_trees = int(annual_emissions_kg / 21.0)     # 1 albero ≈ 21 kg CO2/anno
    equivalent_km_car = int(annual_emissions_kg / 0.120)   # 1 km ≈ 120 g CO2

    return {
        "emission_factor": emission_factor,
        "annual_emissions_kg": annual_emissions_kg,
        "annual_emissions_ton": annual_emissions_ton,
        "equivalent_trees": equivalent_trees,
        "equivalent_km_car": equivalent_km_car,
    }


def calculate_savings(
    baseline_emissions_kg: float,
    scenario_emissions_kg: float,
) -> Dict[str, float]:
    """Calcola risparmio tra baseline e scenario."""
    absolute = baseline_emissions_kg - scenario_emissions_kg
    perc = (absolute / baseline_emissions_kg * 100.0) if baseline_emissions_kg > 0 else 0.0

    return {
        "absolute_kg_co2eq": round(absolute, 2),
        "absolute_ton_co2eq": round(absolute / 1000.0, 3),
        "percentage": round(perc, 1),
    }


# ==============================================================================
#                      LETTURA SYSTEM CSV & TABELLA CO₂
# ==============================================================================

def extract_scenario_name_from_system_path(path: Path) -> str:
    """
    Esempio di path:
    results/ecm/heating/test-cy__baseline___users_..._system.csv
    -> "baseline"

    Si prende la seconda parte dopo il primo "__".
    """
    stem = path.stem  # senza ".csv"
    parts = stem.split("__")
    if len(parts) >= 2:
        return parts[1]
    return stem


def compute_annual_system_energy(system_csv_paths: List[Path]) -> pd.DataFrame:
    """
    Legge i CSV del system heating e calcola consumi annuali:

    - QH_gen_out(kWh)
    - QHW_gen_out(kWh)
    - total_gen_kWh = QH + QHW

    Ritorna un DataFrame con una riga per scenario.
    """
    rows: List[Dict[str, Any]] = []

    for p in system_csv_paths:
        df = pd.read_csv(p, sep=",")
        # opzionale: se hai colonna timestamp e vuoi sicurezza sulle date:
        # df["timestamp"] = pd.to_datetime(df["timestamp"])

        scen_name = extract_scenario_name_from_system_path(p)

        qh = df["QH_gen_out(kWh)"].sum() if "QH_gen_out(kWh)" in df.columns else 0.0
        qhw = df["QHW_gen_out(kWh)"].sum() if "QHW_gen_out(kWh)" in df.columns else 0.0

        rows.append(
            {
                "scenario_name": scen_name,
                "system_csv": str(p),
                "QH_gen_out_kWh": qh,
                "QHW_gen_out_kWh": qhw,
                "total_gen_kWh": qh + qhw,
            }
        )

    return pd.DataFrame(rows)


def build_co2_table_from_system(
    system_csv_paths: List[Path],
    default_energy_source: EnergySource = EnergySource.HEAT_PUMP_ELECTRIC,
    country: str = "IT",
) -> pd.DataFrame:
    """
    Usa i consumi annuali del system per calcolare emissioni CO2eq e risparmi
    rispetto al primo scenario (considerato baseline).
    """
    df_energy = compute_annual_system_energy(system_csv_paths)

    # Calcolo emissioni per scenario
    emission_rows: List[Dict[str, Any]] = []
    for _, row in df_energy.iterrows():
        res = calculate_emissions(
            energy_source=default_energy_source,
            annual_consumption_kwh=row["total_gen_kWh"],
            country=country,
        )
        emission_rows.append(
            {
                "scenario_name": row["scenario_name"],
                # "system_csv": row["system_csv"],
                "energy_source": default_energy_source.value,
                # "total_gen_kWh": row["total_gen_kWh"],
                "QH_gen_out_kWh": row["QH_gen_out_kWh"],
                "QHW_gen_out_kWh": row["QHW_gen_out_kWh"],
                "emission_factor_kg_per_kWh": res["emission_factor"],
                "emissions_kg_co2eq": res["annual_emissions_kg"],
                "emissions_ton_co2eq": res["annual_emissions_ton"],
                "equivalent_trees": res["equivalent_trees"],
                "equivalent_km_car": res["equivalent_km_car"],
            }
        )

    df_co2 = pd.DataFrame(emission_rows)

    # Ordina per scenario_name (baseline, wall, wall_window, ...)
    df_co2 = df_co2.sort_values("scenario_name").reset_index(drop=True)

    # Calcolo risparmi rispetto al baseline (prima riga dopo ordinamento)
    if not df_co2.empty:
        baseline_kg = df_co2.loc[0, "emissions_kg_co2eq"]
        savings_abs = []
        savings_ton = []
        savings_perc = []
        for _, r in df_co2.iterrows():
            s = calculate_savings(baseline_kg, r["emissions_kg_co2eq"])
            savings_abs.append(s["absolute_kg_co2eq"])
            savings_ton.append(s["absolute_ton_co2eq"])
            savings_perc.append(s["percentage"])

        df_co2["saving_vs_baseline_kg"] = savings_abs
        df_co2["saving_vs_baseline_ton"] = savings_ton
        df_co2["saving_vs_baseline_%"] = savings_perc

    return df_co2


# ==============================================================================
#                      GRAFICI – MULTI SIMULATION REPORT
# ==============================================================================

def annual_total_HC_sum_chart(
    labels: List[str],
    values_all_sims: List[List[float]],
    sim_labels: List[str],
    theme_type: str = ThemeType.ROMA,
):
    """
    Bar chart con consumi annuali per Q_H e Q_C per N simulazioni.
    """
    c = Bar(init_opts=opts.InitOpts(theme=theme_type)).add_xaxis(labels)

    for vals, lab in zip(values_all_sims, sim_labels):
        c = c.add_yaxis(lab, vals)

    c = c.set_global_opts(
        title_opts=opts.TitleOpts(
            title="Annual heating (Q_H) and cooling (Q_C) consumption",
            subtitle="Values normalized per surface [kWh/m²]",
        ),
        datazoom_opts=[
            opts.DataZoomOpts(range_start=0, range_end=100),
            opts.DataZoomOpts(type_="inside"),
        ],
        toolbox_opts=opts.ToolboxOpts(
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    title="Download as Image"
                ),
                restore=opts.ToolBoxFeatureRestoreOpts(title="Restore"),
                data_view=opts.ToolBoxFeatureDataViewOpts(
                    title="View Data", lang=["Data View", "Close", "Refresh"]
                ),
                data_zoom=opts.ToolBoxFeatureDataZoomOpts(
                    zoom_title="Zoom In", back_title="Zoom Out"
                ),
                magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
            )
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        yaxis_opts=opts.AxisOpts(
            name="kWh/m²",
            axislabel_opts=opts.LabelOpts(formatter="{value}"),
        ),
    )

    c.height = "600px"
    c.width = "1200px"
    return c


class MultiSimulationsReport:
    """
    Confronto tra N simulazioni (building results) e,
    opzionalmente, uso dei system CSV per la tabella CO₂.
    """

    def __init__(
        self,
        dfs: List[pd.DataFrame],
        labels: List[str],
        building_areas: List[float],
    ):
        assert len(dfs) == len(labels) == len(building_areas), \
            "dfs, labels e building_areas devono avere la stessa lunghezza"

        self.dfs = [df.sort_index(axis=0) for df in dfs]
        self.labels = labels
        self.areas = building_areas

    # -------- Helpers per Q_H / Q_C --------

    @staticmethod
    def _monthly_HC(df: pd.DataFrame, unit='W') -> pd.DataFrame:
        df_HC = df.loc[:, ["Q_H", "Q_C"]]
        try:
            df_monthly = df_HC.resample("ME").sum()
        except Exception:
            df_monthly = df_HC.resample("M").sum()
        if unit == 'W':
            df_monthly = df_monthly / 1000.0
        return df_monthly

    @staticmethod
    def _yearly_HC(df: pd.DataFrame, unit='W') -> pd.DataFrame:
        df_HC = df.loc[:, ["Q_H", "Q_C"]]
        try:
            df_yearly = df_HC.resample("YE").sum()
        except Exception:
            df_yearly = df_HC.resample("Y").sum()
        if unit == 'W':
            df_yearly = df_yearly / 1000.0
        return df_yearly

    # -------- Grafici building --------

    def monthly_heating_comparison(
        self,
        folder_directory: str,
        name_file: str = "monthly_heating_comparison",
    ):
        df_monthly_list = [self._monthly_HC(df, unit='W') for df in self.dfs]

        y_data = []
        y_name = []
        for lab, df_m in zip(self.labels, df_monthly_list):
            y_data.append(df_m["Q_H"].tolist())
            y_name.append(f"{lab} - Heating")

        theme_type = ThemeType.ESSOS
        Chart = bar_chart_single(y_name, y_data, theme_type)

        file_path = f"{folder_directory}/{name_file}.html"
        Chart.render(file_path)
        return Chart

    def monthly_cooling_comparison(
        self,
        folder_directory: str,
        name_file: str = "monthly_cooling_comparison",
    ):
        df_monthly_list = [self._monthly_HC(df, unit='W') for df in self.dfs]

        y_data = []
        y_name = []
        for lab, df_m in zip(self.labels, df_monthly_list):
            y_data.append(df_m["Q_C"].tolist())
            y_name.append(f"{lab} - Cooling")

        theme_type = ThemeType.WALDEN
        Chart = bar_chart_single(y_name, y_data, theme_type)

        file_path = f"{folder_directory}/{name_file}.html"
        Chart.render(file_path)
        return Chart

    def annual_total_HC_comparison(
        self,
        folder_directory: str,
        name_file: str = "annual_total_HC_comparison",
    ):
        df_yearly_list = [self._yearly_HC(df, unit=None) for df in self.dfs]

        values_all_sims = []
        for df_y, area in zip(df_yearly_list, self.areas):
            qh_kwh_m2 = df_y["Q_H"].values[0] / (1000.0 * area)
            qc_kwh_m2 = df_y["Q_C"].values[0] / (1000.0 * area)
            values_all_sims.append([round(qh_kwh_m2, 2), round(qc_kwh_m2, 2)])

        labels = ["Q_H", "Q_C"]

        Chart = annual_total_HC_sum_chart(
            labels=labels,
            values_all_sims=values_all_sims,
            sim_labels=self.labels,
            theme_type=ThemeType.ROMA,
        )

        file_path = f"{folder_directory}/{name_file}.html"
        Chart.render(file_path)
        return Chart

    # -------- Pagina completa grafici --------

    def comparison_page(
        self,
        folder_directory: str,
        name_file: str = "comparison_report",
    ):
        """
        HTML con:
        - monthly heating
        - monthly cooling
        - annual Q_H + Q_C [kWh/m²]
        """
        page = Page(layout=Page.SimplePageLayout)
        page.add(
            self.monthly_heating_comparison(
                folder_directory=folder_directory,
                name_file="monthly_heating_comparison",
            ),
            self.monthly_cooling_comparison(
                folder_directory=folder_directory,
                name_file="monthly_cooling_comparison",
            ),
            self.annual_total_HC_comparison(
                folder_directory=folder_directory,
                name_file="annual_total_HC_comparison",
            ),
        )

        file_path = f"{folder_directory}/{name_file}.html"
        page.render(file_path)
        print(f"[OK] Comparison report created: {file_path}")
        return file_path


# ==============================================================================
#                      FUNZIONE HIGH-LEVEL: CSV → TABELLE + GRAFICI
# ==============================================================================
def save_pretty_co2_html_table(
    df: pd.DataFrame,
    path: Path,
    title: str = "CO₂ emissions summary",
    subtitle: str = "Annual system energy & CO₂ savings per scenario",
) -> None:
    """
    Salva una tabella HTML carina con:
    - prima colonna sticky/fissa
    - scroll orizzontale sulle altre colonne
    - stile moderno
    """

    # ------ Format numeri ------
    df_disp = df.copy()
    num_cols = [
        "total_gen_kWh",
        "QH_gen_out_kWh",
        "QHW_gen_out_kWh",
        "emission_factor_kg_per_kWh",
        "emissions_kg_co2eq",
        "emissions_ton_co2eq",
        "saving_vs_baseline_kg",
        "saving_vs_baseline_ton",
        "saving_vs_baseline_%",
    ]
    for col in num_cols:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(
                lambda x: f"{x:,.2f}".replace(",", " ") if isinstance(x, (float, int)) else x
            )

    # ------ HTML della tabella ------
    table_html = df_disp.to_html(
        index=False,
        border=0,
        classes="co2-table",
        escape=False,
    )

    # ------ FULL HTML con stile ------
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
:root {{
  color-scheme: light dark;
}}
body {{
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI";
  margin: 0;
  padding: 2rem;
  background: #f5f5f7;
  color: #222;
}}

.wrapper {{
  max-width: 96%;
  margin: 0 auto;
  background: #ffffff;
  border-radius: 14px;
  box-shadow: 0 12px 35px rgba(0,0,0,0.13);
  padding: 2rem;
}}

h1 {{
  margin: 0;
  font-size: 1.8rem;
}}

.subtitle {{
  margin-top: 0.25rem;
  margin-bottom: 1.4rem;
  color: #6b7280;
  font-size: 0.95rem;
}}

.table-container {{
  overflow-x: auto;
  padding-bottom: 1rem;
}}

.co2-table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 0.92rem;
  min-width: max-content;
}}

.co2-table thead {{
  background: linear-gradient(90deg, #0f766e, #22c55e);
  color: white;
}}

.co2-table th, .co2-table td {{
  padding: 8px 12px;
  border-bottom: 1px solid #e5e7eb;
  white-space: nowrap;
}}

.co2-table tbody tr:nth-child(odd) {{
  background-color: #f9fafb;
}}

.co2-table tbody tr:hover {{
  background-color: #ecfdf5;
  transition: background-color 0.15s ease-out;
}}

/* HEADER sticky */
.co2-table thead th {{
  position: sticky;
  top: 0;
  z-index: 3;
}}

/* PRIMA COLONNA sticky */
.co2-table th:first-child,
.co2-table td:first-child {{
  position: sticky;
  left: 0;
  background: white;
  z-index: 4;
  font-weight: 600;
  border-right: 1px solid #e5e7eb;
}}

</style>
</head>
<body>
<div class="wrapper">
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>

  <div class="table-container">
    {table_html}
  </div>
</div>
</body>
</html>
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(full_html, encoding="utf-8")



def create_reports_from_csv(
    building_csv_paths: List[Path],
    system_csv_paths: List[Path],
    folder_directory: str,
    name_file: str,
    building_areas: List[float],
    labels: Optional[List[str]] = None,
    datetime_column: Optional[str] = None,
    sep: str = ",",
    country: str = "IT",
    energy_source_for_system: EnergySource = EnergySource.HEAT_PUMP_ELECTRIC,
):
    """
    Fa tutto:
      1) Legge i CSV building (results/ecm/*.csv)
      2) Genera grafici di confronto (building) con pyecharts
      3) Legge i CSV system heating (results/ecm/heating/*.csv)
      4) Calcola tabella CO2 e la salva in CSV + HTML
    """

    assert len(building_csv_paths) == len(building_areas), \
        "building_csv_paths e building_areas devono avere la stessa lunghezza"

    if labels is None:
        labels = [f"Simulation {i+1}" for i in range(len(building_csv_paths))]

    assert len(labels) == len(building_csv_paths), \
        "labels deve avere la stessa lunghezza di building_csv_paths"

    # --- 1) Building: lettura DataFrame ---
    dfs_building: List[pd.DataFrame] = []
    for path in building_csv_paths:
        if datetime_column is None:
            df = pd.read_csv(path, index_col=0, parse_dates=True, sep=sep)
        else:
            df = pd.read_csv(path, sep=sep)
            df[datetime_column] = pd.to_datetime(df[datetime_column])
            df = df.set_index(datetime_column)
        dfs_building.append(df)

    # --- 2) Genera grafici building ---
    report = MultiSimulationsReport(
        dfs=dfs_building,
        labels=labels,
        building_areas=building_areas,
    )
    report_html = report.comparison_page(
        folder_directory=folder_directory,
        name_file=name_file,
    )

    # --- 3) System heating → tabella CO₂ ---
    df_co2 = build_co2_table_from_system(
        system_csv_paths=system_csv_paths,
        default_energy_source=energy_source_for_system,
        country=country,
    )

    co2_csv_path = Path(folder_directory) / f"{name_file}_co2_summary.csv"
    co2_html_path = Path(folder_directory) / f"{name_file}_co2_summary.html"

    # CSV “grezzo”
    df_co2.to_csv(co2_csv_path, index=False)

    # HTML carino
    save_pretty_co2_html_table(
        df=df_co2,
        path=co2_html_path,
        title="CO₂ emissions & savings – heating system",
        subtitle="Based on annual generator energy (QH_gen_out + QHW_gen_out)",
    )

    print(f"[OK] CO2 summary CSV:  {co2_csv_path}")
    print(f"[OK] CO2 summary HTML: {co2_html_path}")

    return {
        "report_html": str(report_html),
        "co2_csv": str(co2_csv_path),
        "co2_html": str(co2_html_path),
    }


# ==============================================================================
#                      ESEMPIO D'USO (MAIN)
# ==============================================================================

if __name__ == "__main__":
    # Cartelle base
    base_folder = Path("results/")
    system_folder = Path("results/ecm/heating/")

    # --- Building CSV: risultati edificio (come nel tuo script originale) ---
    building_csv_paths = [
        p for p in base_folder.glob("*.csv")
        if "annual" not in p.name.lower()
    ]
    building_csv_paths = sorted(building_csv_paths)

    if not building_csv_paths:
        raise SystemExit(f"Nessun CSV building trovato in {base_folder}")

    # Building Area
    building_areas = [BUI["building"]["net_floor_area"]] * len(building_csv_paths)

    # Label per le simulazioni (puoi cambiarle)
    labels = [p.stem for p in building_csv_paths]

    # --- System CSV: risultati sistema heating (results/ecm/heating) ---
    system_csv_paths = sorted(system_folder.glob("*.csv"))

    if not system_csv_paths:
        raise SystemExit(f"Nessun CSV system heating trovato in {system_folder}")

    # Ora lanciamo tutto assieme:
    create_reports_from_csv(
        building_csv_paths=building_csv_paths,
        system_csv_paths=system_csv_paths,
        folder_directory=str(base_folder),
        name_file="comparison_simulations_ecm",
        building_areas=building_areas,
        labels=labels,
        datetime_column=None,
        sep=",",
        country="IT",
        energy_source_for_system=EnergySource.NATURAL_GAS,  # o NATURAL_GAS, ecc.
    )
