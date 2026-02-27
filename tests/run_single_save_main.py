from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def _load_building_archetypes() -> List[Dict[str, Any]]:
    ns = runpy.run_path(str(SRC_DIR / "relife_forecasting" / "building_examples.py"))
    value = ns.get("BUILDING_ARCHETYPES")
    if not isinstance(value, list):
        raise RuntimeError("BUILDING_ARCHETYPES non trovato o non valido in building_examples.py.")
    return value


def _list_archetypes() -> List[Dict[str, str]]:
    building_archetypes = _load_building_archetypes()
    out: List[Dict[str, str]] = []
    for item in building_archetypes:
        name = item.get("name")
        category = item.get("category")
        country = item.get("country")
        if isinstance(name, str) and isinstance(category, str) and isinstance(country, str):
            out.append({"name": name, "category": category, "country": country})
    return out


def _pick_archetype(
    *,
    name: str | None,
    category: str | None,
    country: str | None,
) -> Dict[str, str]:
    archetypes = _list_archetypes()
    matches = [
        a
        for a in archetypes
        if (name is None or a["name"] == name)
        and (category is None or a["category"] == category)
        and (country is None or a["country"] == country)
    ]
    if not matches:
        raise ValueError(
            "Nessun archetipo trovato con i filtri richiesti. "
            "Usa --list-archetypes per vedere i valori disponibili."
        )
    return matches[0]


def _install_mock_iso(main_module: Any) -> None:
    def _fake_iso_calc(
        bui: Dict[str, Any],
        weather_source: str = "pvgis",
        sankey_graph: bool = False,
        path_weather_file: str | None = None,
    ):
        _ = bui, weather_source, sankey_graph, path_weather_file
        hourly = [
            {"Q_H_ideal": 1200.0, "Q_C_ideal": 150.0},
            {"Q_H_ideal": 1100.0, "Q_C_ideal": 120.0},
        ]
        annual = [{"Q_H_annual": 2300.0, "Q_C_annual": 270.0}]
        return hourly, annual

    main_module.pybui.ISO52016.Temperature_and_Energy_needs_calculation = _fake_iso_calc


def _load_runtime_deps():
    try:
        from fastapi.testclient import TestClient  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Dipendenza mancante: fastapi. Installa le dipendenze del progetto prima di eseguire questo script."
        ) from exc

    try:
        import relife_forecasting.main as main_module  # noqa: WPS433
    except Exception as exc:
        raise RuntimeError(
            "Impossibile importare relife_forecasting.main. Verifica che le dipendenze runtime (es. uvicorn/fastapi) siano installate."
        ) from exc

    return TestClient, main_module


def _pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in columns:
            return col
    lower_map = {col.lower(): col for col in columns}
    for col in candidates:
        key = col.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _plot_run_single_results(payload: Dict[str, Any], plot_dir: Path) -> List[Path]:
    try:
        import matplotlib
        import pandas as pd

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Dipendenze plotting mancanti: installa matplotlib e pandas per generare i grafici."
        ) from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: List[Path] = []

    results = payload.get("results")
    if not isinstance(results, list):
        return saved_plots

    successful_results = [item for item in results if isinstance(item, dict) and item.get("status") == "success"]
    if not successful_results:
        return saved_plots

    hourly_series: List[Dict[str, Any]] = []
    annual_rows: List[Dict[str, float]] = []

    for item in successful_results:
        scenario_id = str(item.get("scenario_id") or item.get("combo_tag") or "scenario")
        files = item.get("files") if isinstance(item.get("files"), dict) else {}

        hourly_csv = files.get("hourly_csv")
        if isinstance(hourly_csv, str) and Path(hourly_csv).exists():
            df_hourly = pd.read_csv(hourly_csv)
            columns = list(df_hourly.columns)
            h_col = _pick_column(columns, ["Q_H_ideal", "Q_H", "QH", "heating"])
            c_col = _pick_column(columns, ["Q_C_ideal", "Q_C", "QC", "cooling"])
            if h_col or c_col:
                hourly_series.append(
                    {
                        "scenario_id": scenario_id,
                        "df": df_hourly,
                        "heating_col": h_col,
                        "cooling_col": c_col,
                    }
                )

        annual_csv = files.get("annual_csv")
        if isinstance(annual_csv, str) and Path(annual_csv).exists():
            df_annual = pd.read_csv(annual_csv)
            if not df_annual.empty:
                columns = list(df_annual.columns)
                h_col = _pick_column(columns, ["Q_H_annual", "Q_H", "QH"])
                c_col = _pick_column(columns, ["Q_C_annual", "Q_C", "QC"])
                row0 = df_annual.iloc[0]
                heating = float(pd.to_numeric(row0.get(h_col), errors="coerce")) if h_col else 0.0
                cooling = float(pd.to_numeric(row0.get(c_col), errors="coerce")) if c_col else 0.0
                annual_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "Q_H_annual": 0.0 if heating != heating else heating,
                        "Q_C_annual": 0.0 if cooling != cooling else cooling,
                    }
                )

    if hourly_series:
        fig, (ax_h, ax_c) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for item in hourly_series:
            df_hourly = item["df"]
            x_axis = range(len(df_hourly))
            if item["heating_col"]:
                y_h = pd.to_numeric(df_hourly[item["heating_col"]], errors="coerce").fillna(0.0)
                ax_h.plot(x_axis, y_h, label=item["scenario_id"])
            if item["cooling_col"]:
                y_c = pd.to_numeric(df_hourly[item["cooling_col"]], errors="coerce").fillna(0.0)
                ax_c.plot(x_axis, y_c, label=item["scenario_id"])

        ax_h.set_title("Hourly Heating Demand")
        ax_h.set_ylabel("Heating")
        ax_h.grid(True, alpha=0.3)
        ax_h.legend()

        ax_c.set_title("Hourly Cooling Demand")
        ax_c.set_xlabel("Hour index")
        ax_c.set_ylabel("Cooling")
        ax_c.grid(True, alpha=0.3)
        ax_c.legend()

        fig.tight_layout()
        hourly_plot_path = plot_dir / "run_single_save_hourly.png"
        fig.savefig(hourly_plot_path, dpi=150)
        plt.close(fig)
        saved_plots.append(hourly_plot_path)

    if annual_rows:
        df_bar = pd.DataFrame(annual_rows).set_index("scenario_id")[["Q_H_annual", "Q_C_annual"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        df_bar.plot(kind="bar", ax=ax)
        ax.set_title("Annual Heating/Cooling by Scenario")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Energy")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        annual_plot_path = plot_dir / "run_single_save_annual.png"
        fig.savefig(annual_plot_path, dpi=150)
        plt.close(fig)
        saved_plots.append(annual_plot_path)

    return saved_plots


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test endpoint /ecm_application/run_single_save usando dati archetipi."
    )
    parser.add_argument("--list-archetypes", action="store_true", help="Stampa tutti gli archetipi e termina.")
    parser.add_argument("--name", default="SFH_1946_1969", help="Nome archetipo.")
    parser.add_argument("--category", default="Single Family House", help="Categoria archetipo.")
    parser.add_argument("--country", default="Austria", help="Paese archetipo.")
    parser.add_argument("--weather-source", default="pvgis", choices=["pvgis", "epw"], help="Fonte meteo.")
    parser.add_argument("--include-baseline", action="store_true", default=True, help="Esegue scenario baseline.")
    parser.add_argument("--no-include-baseline", dest="include_baseline", action="store_false")
    parser.add_argument("--u-wall", type=float, default=None, help="Nuovo U-value parete.")
    parser.add_argument("--u-roof", type=float, default=None, help="Nuovo U-value tetto.")
    parser.add_argument("--u-window", type=float, default=None, help="Nuovo U-value finestra.")
    parser.add_argument("--u-slab", type=float, default=None, help="Nuovo U-value solaio.")
    parser.add_argument("--use-heat-pump", action="store_true", default=False, help="Attiva heat pump.")
    parser.add_argument("--heat-pump-cop", type=float, default=3.2, help="COP heat pump.")
    parser.add_argument("--use-pv", action="store_true", default=False, help="Attiva PV integrato.")
    parser.add_argument("--pv-kwp", type=float, default=None, help="Potenza PV [kWp]. Richiesta con --use-pv.")
    parser.add_argument("--output-dir", default="results/ecm_api_single_script", help="Cartella CSV output.")
    parser.add_argument("--bui-dir", default="building_examples_ecm_api_single_script", help="Cartella BUI output.")
    parser.add_argument("--save-bui", action="store_true", default=True, help="Salva BUI modificati.")
    parser.add_argument("--no-save-bui", dest="save_bui", action="store_false")
    parser.add_argument("--plot", action="store_true", default=True, help="Genera grafici PNG dai CSV restituiti.")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disattiva generazione grafici.")
    parser.add_argument("--plot-dir", default=None, help="Cartella output grafici (default: <output-dir>/plots).")
    parser.add_argument(
        "--real-iso",
        action="store_true",
        default=False,
        help="Usa la simulazione ISO reale. Di default usa mock offline.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_archetypes:
        for item in _list_archetypes():
            print(f"{item['country']} | {item['category']} | {item['name']}")
        return 0

    try:
        selected = _pick_archetype(name=args.name, category=args.category, country=args.country)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        test_client_cls, main_module = _load_runtime_deps()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not args.real_iso:
        _install_mock_iso(main_module)

    params: Dict[str, Any] = {
        "archetype": "true",
        "category": selected["category"],
        "country": selected["country"],
        "name": selected["name"],
        "weather_source": args.weather_source,
        "include_baseline": str(args.include_baseline).lower(),
        "use_heat_pump": str(args.use_heat_pump).lower(),
        "heat_pump_cop": args.heat_pump_cop,
        "use_pv": str(args.use_pv).lower(),
        "output_dir": args.output_dir,
        "save_bui": str(args.save_bui).lower(),
        "bui_dir": args.bui_dir,
    }

    if args.u_wall is not None:
        params["u_wall"] = args.u_wall
    if args.u_roof is not None:
        params["u_roof"] = args.u_roof
    if args.u_window is not None:
        params["u_window"] = args.u_window
    if args.u_slab is not None:
        params["u_slab"] = args.u_slab
    if args.use_pv:
        if args.pv_kwp is None:
            print("Con --use-pv devi passare anche --pv-kwp > 0.", file=sys.stderr)
            return 2
        params["pv_kwp"] = args.pv_kwp

    client = test_client_cls(main_module.app)
    response = client.post("/ecm_application/run_single_save", params=params)

    print(f"HTTP {response.status_code}")
    try:
        payload = response.json()
    except Exception:
        print(response.text)
        return 1

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if response.status_code != 200:
        return 1

    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else Path(args.output_dir) / "plots"
        try:
            plots = _plot_run_single_results(payload=payload, plot_dir=plot_dir)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        if plots:
            print("Grafici salvati:")
            for plot_path in plots:
                print(f"- {plot_path}")
        else:
            print("Nessun grafico generato (dati non disponibili nei CSV).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
