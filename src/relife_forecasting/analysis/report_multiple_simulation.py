from pyecharts.globals import ThemeType
from pyecharts.charts import Page, Bar, Line
from pyecharts import options as opts
from relife_forecasting.utils.graphs import *
import pandas as pd
import json
from pathlib import Path
import os
import pytest

from relife_forecasting.building_examples import BUI

# -------------------------------------------------------------------------------------------------
#   ANNUAL CHART FOR N SIMULATIONS (BUILDING)
# -------------------------------------------------------------------------------------------------

def annual_total_HC_sum_chart(
    labels: list,
    values_all_sims: list[list[float]],
    sim_labels: list[str],
    theme_type: str = ThemeType.ROMA,
):
    """
    Bar chart with annual Q_H and Q_C for N simulations.

    :param labels: x-axis labels (e.g., ["Q_H", "Q_C"])
    :param values_all_sims: list of simulation values:
                            [
                                [qh1, qc1],   # sim 1
                                [qh2, qc2],   # sim 2
                                [qh3, qc3],   # sim 3
                                ...
                            ]
    :param sim_labels: labels for each simulation (same length as values_all_sims)
    :param theme_type: pyecharts theme
    """
    c = (
        Bar(init_opts=opts.InitOpts(theme=theme_type))
        .add_xaxis(labels)
    )

    for vals, lab in zip(values_all_sims, sim_labels):
        c = c.add_yaxis(lab, vals)

    c = c.set_global_opts(
        title_opts=opts.TitleOpts(
            title="Annual heating [kWh/m²]",
            # subtitle="normalized per surface ",
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


# -------------------------------------------------------------------------------------------------
#   MONTHLY TIME SERIES GENERATION CHART (SYSTEM)
# -------------------------------------------------------------------------------------------------

def monthly_generation_time_series_chart(
    x_labels: list[str],
    series_qh: list[list[float]],
    series_qhw: list[list[float]],
    sim_labels: list[str],
    title: str = "Monthly generator energy",
    theme_type: str = ThemeType.ROMA,
):
    """
    Monthly line chart for:
      - QH_gen_out(kWh)
      - QHW_gen_out(kWh)
    across all simulations.

    :param x_labels: x-axis labels (e.g., ["2009-01", "2009-02", ...])
    :param series_qh: list of monthly QH_gen_out(kWh) series
    :param series_qhw: list of monthly QHW_gen_out(kWh) series
    :param sim_labels: simulation labels
    :param title: chart title
    :param theme_type: pyecharts theme
    """
    c = Line(init_opts=opts.InitOpts(theme=theme_type))
    c.add_xaxis(x_labels)

    for qh_vals, qhw_vals, lab in zip(series_qh, series_qhw, sim_labels):
        # One line per heating
        # c.add_yaxis(
        #     f"{lab} - QH_gen_out",
        #     qh_vals,
        #     is_smooth=True,
        #     linestyle_opts=opts.LineStyleOpts(width=2),
        # )
        # One line per DHW (QHW)
        c.add_yaxis(
            f"{lab} - QHW_gen_out",
            qhw_vals,
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2, type_="dashed"),
        )

    c.set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            subtitle="generator output energy (heating)",
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
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
        yaxis_opts=opts.AxisOpts(
            name="kWh",
            axislabel_opts=opts.LabelOpts(formatter="{value}"),
        ),
        xaxis_opts=opts.AxisOpts(
            name="Month",
            axislabel_opts=opts.LabelOpts(rotate=45),
        ),
    )

    c.height = "600px"
    c.width = "1200px"
    return c


# -------------------------------------------------------------------------------------------------
#   REPORT FOR N SIMULATIONS
# -------------------------------------------------------------------------------------------------

class MultiSimulationsReport:
    """
    Class to build a comparison report across N simulations
    from N DataFrames (e.g., read from CSV).

    Each "building" DataFrame is expected to have:
        - datetime index
        - columns: 'Q_H' (heating), 'Q_C' (cooling),
          optionally 'Q_HC', 'T_op', 'T_ext'

    Each "system" DataFrame (if provided) is expected to have:
        - datetime index
        - columns: 'QH_gen_out(kWh)', 'QHW_gen_out(kWh)' (if available)
    """

    def __init__(
        self,
        building_dfs: list[pd.DataFrame],
        labels: list[str],
        building_areas: list[float],
        system_dfs: list[pd.DataFrame] | None = None,
    ):
        assert len(building_dfs) == len(labels) == len(building_areas), (
            "building_dfs, labels, and building_areas must have the same length"
        )

        self.dfs = [df.sort_index(axis=0) for df in building_dfs]
        self.labels = labels
        self.areas = building_areas

        # System (optional)
        if system_dfs is not None:
            assert len(system_dfs) == len(building_dfs), (
                "system_dfs must have the same length as building_dfs"
            )
            self.system_dfs = [df.sort_index(axis=0) for df in system_dfs]
        else:
            self.system_dfs = []

    # ------------------------ HELPERS ------------------------

    @staticmethod
    def _monthly_HC(df: pd.DataFrame, unit='W') -> pd.DataFrame:
        """
        Returns a DataFrame with monthly sum of Q_H and Q_C.
        """
        df_HC = df.loc[:, ["Q_H", "Q_C"]]
        try:
            df_monthly = df_HC.resample("ME").sum()
        except Exception:
            df_monthly = df_HC.resample("M").sum()
        if unit == 'W':
            df_monthly = df_monthly / 1000
        return df_monthly

    @staticmethod
    def _yearly_HC(df: pd.DataFrame, unit='W') -> pd.DataFrame:
        """
        Returns a DataFrame with annual sum of Q_H and Q_C.
        """
        df_HC = df.loc[:, ["Q_H", "Q_C"]]
        try:
            df_yearly = df_HC.resample("YE").sum()
        except Exception:
            df_yearly = df_HC.resample("Y").sum()
        if unit == 'W':
            df_yearly = df_yearly / 1000
        return df_yearly

    @staticmethod
    def _monthly_gen_energy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with monthly sums of:
          - QH_gen_out(kWh)
          - QHW_gen_out(kWh)
        if present in the DataFrame.

        The index must be datetime.
        """
        cols = []
        if "QH_gen_out(kWh)" in df.columns:
            cols.append("QH_gen_out(kWh)")
        if "QHW_gen_out(kWh)" in df.columns:
            cols.append("QHW_gen_out(kWh)")

        if not cols:
            raise ValueError(
                "The DataFrame does not contain 'QH_gen_out(kWh)' or 'QHW_gen_out(kWh)'."
            )

        df_gen = df.loc[:, cols]

        try:
            df_monthly = df_gen.resample("ME").sum()
        except Exception:
            df_monthly = df_gen.resample("M").sum()

        return df_monthly

    # ------------------------ COMPARISON CHARTS (BUILDING) ------------------------

    def monthly_heating_comparison(
        self,
        folder_directory: str,
        name_file: str = "monthly_heating_comparison",
    ):
        """
        Monthly heating consumption comparison across N simulations.
        Values are in kWh.
        """

        df_monthly_list = [self._monthly_HC(df, unit='W') for df in self.dfs]

        y_data = []
        y_name = []
        for lab, df_m in zip(self.labels, df_monthly_list):
            y_data.append([v for v in df_m["Q_H"].tolist()])
            y_name.append(f"{lab} - Heating")

        theme_type = ThemeType.ESSOS
        Chart = bar_chart_single(y_name, y_data, theme_type)

        file_path = "{}/{}.html".format(folder_directory, name_file)
        Chart.render(file_path)
        return Chart

    def monthly_cooling_comparison(
        self,
        folder_directory: str,
        name_file: str = "monthly_cooling_comparison",
    ):
        """
        Monthly cooling consumption comparison across N simulations.
        Values are in kWh.
        """

        df_monthly_list = [self._monthly_HC(df, unit='W') for df in self.dfs]

        y_data = []
        y_name = []
        for lab, df_m in zip(self.labels, df_monthly_list):
            y_data.append([v for v in df_m["Q_C"].tolist()])
            y_name.append(f"{lab} - Cooling")

        theme_type = ThemeType.WALDEN
        Chart = bar_chart_single(y_name, y_data, theme_type)

        file_path = "{}/{}.html".format(folder_directory, name_file)
        Chart.render(file_path)
        return Chart

    # ------------------------ COMPARISON CHART (SYSTEM) ------------------------

    def monthly_generation_comparison(
        self,
        folder_directory: str,
        name_file: str = "monthly_generation_comparison",
    ):
        """
        Monthly line chart for QH_gen_out(kWh) and QHW_gen_out(kWh)
        across all system simulations (all CSVs in results/ecm/heating).
        """
        if not self.system_dfs:
            # No system data provided -> no chart
            raise RuntimeError(
                "No system DataFrame provided to MultiSimulationsReport (system_dfs is empty)."
            )

        monthly_list = [self._monthly_gen_energy(df) for df in self.system_dfs]

        # X-axis labels as "YYYY-MM" (use the first simulation index)
        first_monthly = monthly_list[0]
        x_labels = [idx.strftime("%Y-%m") for idx in first_monthly.index]

        series_qh = []
        series_qhw = []

        for df_m in monthly_list:
            if "QH_gen_out(kWh)" in df_m.columns:
                qh_vals = df_m["QH_gen_out(kWh)"].tolist()
            else:
                qh_vals = [0.0] * len(df_m)

            if "QHW_gen_out(kWh)" in df_m.columns:
                qhw_vals = df_m["QHW_gen_out(kWh)"].tolist()
            else:
                qhw_vals = [0.0] * len(df_m)

            series_qh.append(qh_vals)
            series_qhw.append(qhw_vals)

        chart_title = "Primary energy - heating"

        Chart = monthly_generation_time_series_chart(
            x_labels=x_labels,
            series_qh=series_qh,
            series_qhw=series_qhw,
            sim_labels=self.labels,
            title=chart_title,
            theme_type=ThemeType.ROMA,
        )

        file_path = f"{folder_directory}/{name_file}.html"
        Chart.render(file_path)
        return Chart

    def annual_total_HC_comparison(
        self,
        folder_directory: str,
        name_file: str = "annual_total_HC_comparison",
    ):
        """
        Annual comparison of heating (Q_H) and cooling (Q_C) consumption,
        normalized per surface [kWh/m²], for all simulations.
        """

        # Use unit=None to avoid division by 1000 inside _yearly_HC
        df_yearly_list = [self._yearly_HC(df, unit=None) for df in self.dfs]

        values_all_sims = []
        for df_y, area in zip(df_yearly_list, self.areas):
            # Q_* in Wh -> /1000 for kWh, then /area [m²]
            qh_kwh_m2 = df_y["Q_H"].values[0] / (1000 * area)
            qc_kwh_m2 = df_y["Q_C"].values[0] / (1000 * area)
            values_all_sims.append([round(qh_kwh_m2, 2), round(qc_kwh_m2, 2)])

        labels = ["Q_H", "Q_C"]  # heating and cooling

        Chart = annual_total_HC_sum_chart(
            labels=labels,
            values_all_sims=values_all_sims,
            sim_labels=self.labels,
            theme_type=ThemeType.ROMA,
        )

        file_path = "{}/{}.html".format(folder_directory, name_file)
        Chart.render(file_path)
        return Chart

    # ------------------------ FULL REPORT ------------------------

    def comparison_page(
        self,
        folder_directory: str,
        name_file: str = "comparison_report",
    ):
        """
        Creates an HTML report comparing all simulations:

        - Monthly generator energy (system, QH_gen_out & QHW_gen_out, all simulations)
        - Monthly heating consumption (bar chart, building)
        - Monthly cooling consumption (bar chart, building)
        - Annual total heating + cooling [kWh/m²] (bar chart, building)
        """

        page = Page(layout=Page.SimplePageLayout)

        # 1) System generation chart (if available)
        if self.system_dfs:
            page.add(
                self.monthly_generation_comparison(
                    folder_directory=folder_directory,
                    name_file="monthly_generation_comparison",
                )
            )

        # 2) Heating / cooling / annual HC (building)
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

        file_path = "{}/{}.html".format(folder_directory, name_file)
        page.render(file_path)

        print("Comparison report created!")
        return json.dumps({"report": "comparison_created"})


# ========================================================================================================
#   HELPER FUNCTION FOR N CSV FILES (BUILDING + SYSTEM)
# ========================================================================================================

def create_multi_simulations_report_from_csv(
    building_csv_paths: list[str],
    folder_directory: str,
    name_file: str,
    building_areas: list[float],
    labels: list[str] | None = None,
    datetime_column: str = None,
    sep: str = ",",
    system_csv_paths: list[str] | None = None,
):
    """
    High-level function that:
        - reads N building CSV files (results/ecm)
        - optionally reads N system CSV files (results/ecm/heating)
        - builds the comparison report
        - saves a final HTML file

    :param building_csv_paths: list of paths to building simulation CSVs
    :param folder_directory: directory where the report is saved
    :param name_file: name of the HTML report file (without extension)
    :param building_areas: list of building areas for the simulations [m²]
    :param labels: list of simulation labels (if None -> Simulation 1, 2, 3, ...)
    :param datetime_column: name of the datetime column if not used as index
    :param sep: CSV separator (default ',')
    :param system_csv_paths: list of paths to system CSVs (same length as building_csv_paths)
    """

    if labels is None:
        labels = [f"Simulation {i+1}" for i in range(len(building_csv_paths))]

    assert len(building_csv_paths) == len(building_areas) == len(labels), (
        "building_csv_paths, building_areas, and labels must have the same length"
    )

    # ---- Building CSV ----
    building_dfs = []
    for path in building_csv_paths:
        if datetime_column is None:
            df = pd.read_csv(path, index_col=0, parse_dates=True, sep=sep)
        else:
            df = pd.read_csv(path, sep=sep)
            df[datetime_column] = pd.to_datetime(df[datetime_column])
            df = df.set_index(datetime_column)
        building_dfs.append(df)

    # ---- System CSV (optional) ----
    system_dfs = None
    if system_csv_paths is not None:
        assert len(system_csv_paths) == len(building_csv_paths), (
            "system_csv_paths must have the same length as building_csv_paths"
        )
        tmp_system_dfs = []
        for path in system_csv_paths:
            if datetime_column is None:
                df_s = pd.read_csv(path, index_col=0, parse_dates=True, sep=sep)
            else:
                df_s = pd.read_csv(path, sep=sep)
                df_s[datetime_column] = pd.to_datetime(df_s[datetime_column])
                df_s = df_s.set_index(datetime_column)
            tmp_system_dfs.append(df_s)
        system_dfs = tmp_system_dfs

    report = MultiSimulationsReport(
        building_dfs=building_dfs,
        labels=labels,
        building_areas=building_areas,
        system_dfs=system_dfs,
    )

    return report.comparison_page(folder_directory=folder_directory, name_file=name_file)


# ========================================================================================================
#   WRAPPER FOR 2 SIMULATIONS (OPTIONAL)
# ========================================================================================================

def create_two_simulations_report_from_csv(
    building_csv_path_1: str,
    building_csv_path_2: str,
    folder_directory: str,
    name_file: str,
    building_area_1: float,
    building_area_2: float,
    label1: str = "Simulation 1",
    label2: str = "Simulation 2",
    datetime_column: str = None,
    sep: str = ",",
    system_csv_path_1: str | None = None,
    system_csv_path_2: str | None = None,
):
    """
    Wrapper that uses the generic N-simulations function.
    """
    system_paths = None
    if system_csv_path_1 is not None and system_csv_path_2 is not None:
        system_paths = [system_csv_path_1, system_csv_path_2]

    return create_multi_simulations_report_from_csv(
        building_csv_paths=[building_csv_path_1, building_csv_path_2],
        folder_directory=folder_directory,
        name_file=name_file,
        building_areas=[building_area_1, building_area_2],
        labels=[label1, label2],
        datetime_column=datetime_column,
        sep=sep,
        system_csv_paths=system_paths,
    )


# ========================================================================================================
#   TEST ENTRY POINT (MULTI-SIM, BUILDING + SYSTEM)
# ========================================================================================================

def test_create_multi_simulations_report_from_csv():
    """
    Smoke test that generates a multi-simulation report when CSVs are available.

    Assumes:
      - building CSVs are in: results/*.csv (excluding "annual")
      - system CSVs are in:   results/ecm/heating/*.csv
      - files are aligned 1:1 in order
    """
    # Folders
    building_folder = "results/"
    system_folder = "results/ecm/heating"

    # --- BUILDING FILES ---
    building_csv_list = [
        str(p) for p in Path(building_folder).glob("*.csv")
        if "annual" not in p.name.lower()
    ]
    building_csv_list = sorted(building_csv_list)

    # --- SYSTEM FILES ---
    system_csv_list = sorted(
        str(p) for p in Path(system_folder).glob("*.csv")
    )

    if not building_csv_list:
        pytest.skip("No building CSVs found for report generation.")

    if not system_csv_list:
        pytest.skip("No system CSVs found for report generation.")

    # Minimal consistency check
    if len(system_csv_list) != len(building_csv_list):
        pytest.skip(
            "Building CSV count differs from system CSV count."
        )

    # Areas: same area for all simulations (example)
    areas_list = [BUI["building"]["net_floor_area"]] * len(building_csv_list)

    # Labels derived from building file names
    labels_list = [
        os.path.basename(p).split("__")[1].split("___")[0]
        for p in building_csv_list
    ]

    create_multi_simulations_report_from_csv(
        building_csv_paths=building_csv_list,
        folder_directory="results",
        name_file="comparison_simulations_with_system",
        building_areas=areas_list,
        labels=labels_list,
        system_csv_paths=system_csv_list,
    )