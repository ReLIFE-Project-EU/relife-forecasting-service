"""
Compare two /simulate runs of the same archetype against a running API:
  1) archetype as stored (no floor_area)
  2) archetype with a different floor_area (geometry is rescaled by the API)

Start the API first (e.g. `uv run uvicorn relife_forecasting.main:app --port 8000`), then:

  python scripts/run_floor_area_comparison.py
  python scripts/run_floor_area_comparison.py --floor-area 187.5 --weather epw \
      --epw src/relife_forecasting/epw_weather/GRC_Athens.167160_IWEC.epw
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

ARCHETYPE = {
    "archetype": "true",
    "category": "Single Family House",
    "country": "Austria",
    "name": "SFH_1946_1969",
}


def simulate(
    client: httpx.Client,
    weather: str,
    epw: Optional[Path],
    floor_area: Optional[float],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {**ARCHETYPE, "weather_source": weather}
    if floor_area is not None:
        params["floor_area"] = floor_area
    files = {"epw_file": (epw.name, epw.read_bytes())} if weather == "epw" else None
    r = client.post("/simulate", params=params, files=files)
    r.raise_for_status()
    return r.json()


def building_area(client: httpx.Client, floor_area: Optional[float]) -> float:
    params: Dict[str, Any] = dict(ARCHETYPE)
    if floor_area is not None:
        params["floor_area"] = floor_area
    r = client.post("/building", params=params)
    r.raise_for_status()
    return r.json()["bui"]["building"]["net_floor_area"]


def summarize(label: str, area: float, result: Dict[str, Any]) -> Dict[str, float]:
    rows = result["results"]["hourly_building"]
    q_h = sum(row.get("Q_H") or 0.0 for row in rows) / 1000.0  # Wh -> kWh
    q_c = sum(row.get("Q_C") or 0.0 for row in rows) / 1000.0
    print(
        f"{label:<18} area={area:7.1f} m2  hours={len(rows)}  "
        f"Q_H={q_h:10.0f} kWh ({q_h / area:6.1f}/m2)  "
        f"Q_C={q_c:10.0f} kWh ({q_c / area:6.1f}/m2)"
    )
    return {"area": area, "Q_H": q_h, "Q_C": q_c}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--floor-area", type=float, default=187.5, help="Second-case net floor area (m2).")
    ap.add_argument("--weather", choices=["pvgis", "epw"], default="pvgis")
    ap.add_argument("--epw", type=Path, default=None, help="EPW file (required with --weather epw).")
    ap.add_argument("--token", default=None, help="Bearer token, if the API requires authentication.")
    args = ap.parse_args()

    if args.weather == "epw" and args.epw is None:
        ap.error("--weather epw requires --epw")

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=600.0) as client:
        base = summarize(
            "archetype",
            building_area(client, None),
            simulate(client, args.weather, args.epw, None),
        )
        new = summarize(
            "modified area",
            building_area(client, args.floor_area),
            simulate(client, args.weather, args.epw, args.floor_area),
        )

    print(
        f"\nchange: area {new['area'] / base['area'] - 1:+.0%}, "
        f"Q_H {new['Q_H'] / base['Q_H'] - 1:+.0%}, "
        f"Q_C {new['Q_C'] / base['Q_C'] - 1:+.0%}"
    )


if __name__ == "__main__":
    main()
