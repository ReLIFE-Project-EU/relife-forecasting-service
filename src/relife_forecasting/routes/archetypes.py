from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

try:
    from relife_forecasting.building_examples import BUILDING_ARCHETYPES
except Exception:
    from building_examples import BUILDING_ARCHETYPES  # type: ignore

router = APIRouter(tags=["archetypes"])


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _surface_summary(surface: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "name": surface.get("name"),
        "type": surface.get("type"),
        "area": surface.get("area"),
        "u_value": surface.get("u_value"),
        "orientation": surface.get("orientation"),
    }
    if "g_value" in surface:
        summary["g_value"] = surface.get("g_value")
    if "thermal_capacity" in surface:
        summary["thermal_capacity"] = surface.get("thermal_capacity")
    if "height" in surface:
        summary["height"] = surface.get("height")
    if "width" in surface:
        summary["width"] = surface.get("width")
    return summary


def _building_summary(archetype: Dict[str, Any]) -> Dict[str, Any]:
    building = archetype.get("bui", {}).get("building", {})
    surfaces = archetype.get("bui", {}).get("building_surface", [])

    selected_surfaces: Dict[str, Dict[str, Any]] = {}
    for surface in surfaces:
        surface_name = str(surface.get("name", "")).lower()
        surface_type = str(surface.get("type", "")).lower()
        orientation = surface.get("orientation") or {}
        tilt = orientation.get("tilt")

        if "roof" in surface_name and "roof" not in selected_surfaces:
            selected_surfaces["roof"] = {"area": surface.get("area")}
        elif "slab to ground" in surface_name and "floor" not in selected_surfaces:
            selected_surfaces["floor"] = {"area": surface.get("area")}
        elif surface_type == "opaque" and tilt == 90 and "wall" not in selected_surfaces:
            selected_surfaces["wall"] = {"area": surface.get("area")}
        elif surface_type == "transparent" and tilt == 90 and "window" not in selected_surfaces:
            selected_surfaces["window"] = {"area": surface.get("area")}

    system = archetype.get("system") or {}
    return {
        "name": archetype.get("name"),
        "country": archetype.get("country"),
        "surfaces": selected_surfaces,
        "generator": {
            "type": system.get("generator_circuit"),
            "power": system.get("full_load_power"),
        },
    }


@router.get("/building/archetypes_info_capex", tags=["archetypes"])
def list_archetypes(
    country: Optional[str] = Query(None, description="Country to filter by, e.g. Austria, Greece, Italy."),
    building_type: Optional[str] = Query(
        None,
        description="Building typology: Single Family House, Multi family House, Apartment buildings.",
    ),
    name: Optional[str] = Query(None, description="Specific archetype name."),
) -> Dict[str, Any]:
    """
    Return archetypes from `building_examples.py` in a readable form.

    Filters are optional. When provided, the endpoint returns only the matching
    archetypes for the requested country and/or building typology.
    """
    matches: List[Dict[str, Any]] = []
    country_filter = _normalize(country) if country else None
    type_filter = _normalize(building_type) if building_type else None
    name_filter = _normalize(name) if name else None

    for archetype in BUILDING_ARCHETYPES:
        if country_filter and _normalize(str(archetype.get("country", ""))) != country_filter:
            continue
        if type_filter and _normalize(str(archetype.get("category", ""))) != type_filter:
            continue
        if name_filter and _normalize(str(archetype.get("name", ""))) != name_filter:
            continue
        matches.append(_building_summary(archetype))

    grouped = OrderedDict()
    for item in matches:
        grouped.setdefault(item["country"], []).append(item)

    if country or building_type:
        return {
            "filters": {"country": country, "building_type": building_type, "name": name},
            "count": len(matches),
            "archetypes": matches,
        }

    return {
        "count": len(matches),
        "countries": list(grouped.keys()),
        "archetypes_by_country": grouped,
    }


@router.get("/building/archetypes/options", tags=["archetypes"])
def list_archetype_options() -> Dict[str, Any]:
    """Return available countries and typologies derived from the archetype list."""
    countries = sorted({str(item.get("country")) for item in BUILDING_ARCHETYPES if item.get("country")})
    building_types = sorted({str(item.get("category")) for item in BUILDING_ARCHETYPES if item.get("category")})
    return {"countries": countries, "building_types": building_types, "count": len(BUILDING_ARCHETYPES)}
