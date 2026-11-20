from enum import StrEnum
from typing import NotRequired, TypedDict


class Country(StrEnum):
    AUSTRIA = "Austria"
    BELGIUM = "Belgium"
    BULGARIA = "Bulgaria"
    CROATIA = "Croatia"
    CYPRUS = "Cyprus"
    CZECH_REPUBLIC = "Czech Republic"
    DENMARK = "Denmark"
    ESTONIA = "Estonia"
    FINLAND = "Finland"
    FRANCE = "France"
    GERMANY = "Germany"
    GREECE = "Greece"
    HUNGARY = "Hungary"
    IRELAND = "Ireland"
    ITALY = "Italy"
    LATVIA = "Latvia"
    LITHUANIA = "Lithuania"
    LUXEMBOURG = "Luxembourg"
    MALTA = "Malta"
    NETHERLANDS = "Netherlands"
    POLAND = "Poland"
    PORTUGAL = "Portugal"
    ROMANIA = "Romania"
    SLOVAKIA = "Slovakia"
    SLOVENIA = "Slovenia"
    SPAIN = "Spain"
    SWEDEN = "Sweden"


class BuildingCategory(StrEnum):
    SINGLE_FAMILY_HOUSE = "Single Family House"
    MULTI_FAMILY_HOUSE = "Multi family House"
    APARTMENT_BUILDINGS = "Apartment buildings"


class BuildingTypeClass(StrEnum):
    DETACHED_HOUSE = "Residential_detached_house"
    APARTMENT = "Residential_apartment"


class ConstructionClass(StrEnum):
    I = "class_i"
    II = "class_ii"
    III = "class_iii"


class ArchetypeSpec(TypedDict):
    """Workbook inputs keyed by country_typology_period sheet name.

    Coordinates are degrees, lengths m, areas m², U-values W/(m² K), and
    surface thermal capacities J/K. Wall area includes windows. Window width
    spans all façades. Floor counts can be fractional. Nonpositive net area
    requests slab area × floors. System efficiencies and factors are ratios.
    """

    country: str
    category: str
    latitude: float
    longitude: float
    exposed_perimeter: float
    wall_thickness: float
    n_floors: float
    building_type_class: str
    net_floor_area: float
    construction_class: str
    roof_area: float
    roof_u: float
    roof_thermal_capacity: float
    wall_area: float
    wall_u: float
    wall_thermal_capacity: float
    slab_area: float
    slab_u: float
    slab_thermal_capacity: float
    window_area: float
    window_u: float
    window_height: float
    window_width: float
    heating_eta_generation: float
    cooling_cop_generation: float
    dhw_eta_generation: float

    g_value: NotRequired[float]  # solar transmittance
    fp_electric: NotRequired[float]  # electric primary-energy factor
    thermal_bridge_delta_u: NotRequired[float]  # W/(m² K)
    infiltration_rate: NotRequired[float]  # BUI air changes/h
    occupants_full_load: NotRequired[float]  # W/m²
    appliances_full_load: NotRequired[float]  # W/m²
    ventilation_rate_1_per_h: NotRequired[float]  # UNI air changes/h
