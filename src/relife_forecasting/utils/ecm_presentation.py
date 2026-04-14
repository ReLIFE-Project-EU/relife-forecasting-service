from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    headers = [_markdown_escape(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        rendered: list[str] = []
        for value in row.tolist():
            if isinstance(value, float):
                rendered.append(_markdown_escape(f"{value:,.2f}"))
            else:
                rendered.append(_markdown_escape(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _format_display_table(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    display = scenario_summary.loc[
        :,
        [
            "scenario_label",
            "iso_total_kwh",
            "primary_total_kwh",
            "final_energy_kwh",
            "annual_co2_ton",
            "co2_saving_pct",
        ],
    ].copy()
    display.columns = [
        "Scenario",
        "ISO total [kWh]",
        "Primary energy [kWh]",
        "Final energy [kWh]",
        "CO2 [t/y]",
        "CO2 saving [%]",
    ]
    for column in display.columns[1:]:
        display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0.0).round(2)
    return display


def _build_findings(scenario_summary: pd.DataFrame) -> list[str]:
    if scenario_summary.empty:
        return ["No successful scenarios were available for the presentation."]

    alternatives = scenario_summary.loc[
        scenario_summary["scenario_name"].astype(str).str.lower() != "baseline"
    ].copy()
    if alternatives.empty:
        return ["Only the baseline scenario is available, so no comparative finding can be derived yet."]

    findings: list[str] = []

    best_co2 = alternatives.loc[alternatives["annual_co2_ton"].idxmin()]
    findings.append(
        f"Lowest annual CO2: {best_co2['scenario_label']} with "
        f"{_safe_float(best_co2['annual_co2_ton']):.2f} tCO2eq/year "
        f"({_safe_float(best_co2['co2_saving_pct']):.1f}% vs baseline)."
    )

    best_primary = alternatives.loc[alternatives["primary_total_kwh"].idxmin()]
    findings.append(
        f"Lowest annual primary energy: {best_primary['scenario_label']} with "
        f"{_safe_float(best_primary['primary_total_kwh']):,.0f} kWh/year."
    )

    best_iso = alternatives.loc[alternatives["iso_total_kwh"].idxmin()]
    findings.append(
        f"Lowest annual ISO 52016 demand: {best_iso['scenario_label']} with "
        f"{_safe_float(best_iso['iso_total_kwh']):,.0f} kWh/year."
    )

    return findings


def _build_considerations() -> list[str]:
    return [
        "CO2 uses a final-energy carrier split: delivered thermal energy is mapped to natural gas, grid imports to grid electricity, and self-consumed PV to solar PV.",
        "Country codes outside the explicit IT/DE mappings fall back to EU emission factors so archetypes such as Greece remain covered.",
        "The comparison is operational and annual: it does not include embodied carbon of retrofit materials or replacement systems.",
        "Results depend on the selected weather source, scenario definitions, and UNI/TS 11300 configuration used during the run.",
    ]


def _ensure_plot_runtime_dirs() -> None:
    cache_root = Path(tempfile.gettempdir()) / "relife_forecasting_plot_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def _render_primary_energy_chart(scenario_summary: pd.DataFrame, output_path: Path) -> None:
    _ensure_plot_runtime_dirs()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = scenario_summary["scenario_label"].tolist()
    iso_values = pd.to_numeric(scenario_summary["iso_total_kwh"], errors="coerce").fillna(0.0)
    primary_values = pd.to_numeric(scenario_summary["primary_total_kwh"], errors="coerce").fillna(0.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = list(range(len(labels)))
    width = 0.36

    ax.bar([pos - width / 2 for pos in x], iso_values, width=width, label="ISO total demand", color="#55A630")
    ax.bar([pos + width / 2 for pos in x], primary_values, width=width, label="Primary energy", color="#007F5F")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("kWh/year")
    ax.set_title("Annual energy results by scenario")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_co2_chart(scenario_summary: pd.DataFrame, output_path: Path) -> None:
    _ensure_plot_runtime_dirs()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = scenario_summary["scenario_label"].tolist()
    emissions = pd.to_numeric(scenario_summary["annual_co2_ton"], errors="coerce").fillna(0.0)
    savings = pd.to_numeric(scenario_summary["co2_saving_pct"], errors="coerce").fillna(0.0)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    bars = ax1.bar(labels, emissions, color=["#007F5F", "#55A630", "#80B918", "#AACC00"][: len(labels)])
    ax1.set_ylabel("tCO2eq/year")
    ax1.set_title("Annual CO2 emissions and reduction vs baseline")
    ax1.grid(axis="y", alpha=0.2)
    ax1.tick_params(axis="x", rotation=18)

    ax2 = ax1.twinx()
    ax2.plot(labels, savings, color="#2B9348", marker="o", linewidth=2.2)
    ax2.set_ylabel("Saving [%]")

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax1.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax2.annotate(
            f"{savings.iloc[idx]:.1f}%",
            xy=(idx, savings.iloc[idx]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#2B9348",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_ecm_presentation(
    *,
    output_path: Union[str, Path],
    report_title: str,
    building_meta: Dict[str, Any],
    run_stats: Dict[str, Any],
    scenario_summary: pd.DataFrame,
    html_report_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pandoc_path = shutil.which("pandoc")
    if pandoc_path is None:
        raise RuntimeError("pandoc is required to generate the PPTX presentation.")

    assets_dir = output_path.parent / f"{output_path.stem}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    energy_chart_path = assets_dir / "annual_energy.png"
    co2_chart_path = assets_dir / "annual_co2.png"
    markdown_path = output_path.with_suffix(".md")

    _render_primary_energy_chart(scenario_summary, energy_chart_path)
    _render_co2_chart(scenario_summary, co2_chart_path)

    display_table = _format_display_table(scenario_summary)
    if len(display_table) > 6:
        display_table = display_table.head(6)

    findings = _build_findings(scenario_summary)
    considerations = _build_considerations()
    scenario_labels = ", ".join(scenario_summary["scenario_label"].astype(str).tolist())

    markdown = f"""# {report_title}

- Building: {building_meta.get("name") or "-"}
- Category: {building_meta.get("category") or "-"}
- Country: {building_meta.get("country") or "-"}
- Weather source: {building_meta.get("weather_source") or "-"}
- Simulations: {int(run_stats.get("total", 0))} total, {int(run_stats.get("successful", 0))} successful, {int(run_stats.get("failed", 0))} failed
- Total runtime: {_safe_float(run_stats.get("total_time")):.2f} s
- Included scenarios: {scenario_labels or "-"}
- Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- HTML report: {str(html_report_path) if html_report_path else "not generated"}

# Process

- The workflow runs a baseline plus the configured renovation scenarios defined in `RENOVATION_SCENARIOS`.
- Each scenario produces hourly ISO 52016 results, annual UNI/TS 11300 primary energy, and scenario-specific CSV exports.
- CO2 is computed with the same core logic used by the API emission endpoints and then aggregated per scenario.
- The presentation focuses on annual scenario deltas so the ranking is easy to read for decision making.

# Generated Data

{_markdown_table(display_table)}

# Energy Results

![]({energy_chart_path.name}){{width=88%}}

- The grouped bars compare annual ISO demand and annual primary energy for each scenario.
- This view highlights whether a retrofit mainly reduces building demand, generation losses, or both.

# CO2 Results

![]({co2_chart_path.name}){{width=88%}}

""" + "\n".join(f"- {finding}" for finding in findings) + """

# Considerations

""" + "\n".join(f"- {item}" for item in considerations) + "\n"

    markdown_path.write_text(markdown, encoding="utf-8")

    command = [
        pandoc_path,
        str(markdown_path),
        "--from",
        "gfm",
        "--to",
        "pptx",
        "--slide-level=1",
        "--resource-path",
        str(assets_dir),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "pandoc failed to generate the presentation: "
            + (completed.stderr.strip() or completed.stdout.strip() or "unknown error")
        )

    return {
        "pptx": str(output_path),
        "markdown": str(markdown_path),
        "energy_chart": str(energy_chart_path),
        "co2_chart": str(co2_chart_path),
    }
