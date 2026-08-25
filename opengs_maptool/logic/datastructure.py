from __future__ import annotations
from typing import TypeAlias

from dataclasses import dataclass
from enum import Enum
from PIL import Image

# Main tab images
LandImage: TypeAlias = Image.Image
BoundaryImage: TypeAlias = Image.Image
DensityImage: TypeAlias = Image.Image
TerrainImage: TypeAlias = Image.Image
TerritoryImage: TypeAlias = Image.Image
ProvinceImage: TypeAlias = Image.Image

# Intermediate data structures
ColorTuple: TypeAlias = tuple[int, int, int]

class RegionLevel(Enum):
    TERRITORY = "territory"
    PROVINCE = "province"

class RegionType(Enum):
    LAND = "land"
    OCEAN = "ocean"
    LAKE = "lake"

@dataclass
class RegionMetadata:
    # For both territories & provinces
    region_level: RegionLevel
    R: int
    G: int
    B: int
    x: float
    y: float
    _pmap_index: int
    territory_id: str | None # used both as territory id & province's parent id

    # Only for provinces
    province_id: str | None
    province_type: RegionType | None # GCE-TODO: use enum
    province_terrain: str | None # GCE-TODO: create terrain enum to use everywhere

    # Only for territories
    territory_type: RegionType | None
    province_ids: list[str] | None


"""
Missing types
- one for all ptype strings

"""

