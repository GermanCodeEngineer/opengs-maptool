"""
Detect fragmented regions from a uint32 region map + metadata.

Output image convention:
- Non-fragmented pixels are rendered in grayscale.
- Fragment pixels are rendered in colorful highlight colors.

The script uses the metadata `seed` field to decide which connected component
is the primary component of a region. Any other component of the same region
is treated as a fragment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing import Any
from PIL import Image
from scipy import ndimage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


FOUR_CONNECTED = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)


REGION_FILE_SPECS = {
    "areas": {
        "image": "cont_areas_image.png",
        "metadata": "cont_areas_data.json",
        "output": "cont_areas_fragments.png",
    },
    "districts": {
        "image": "district_image.png",
        "metadata": "district_data.json",
        "output": "district_fragments.png",
    },
    "territories": {
        "image": "territory_image.png",
        "metadata": "territory_data.json",
        "output": "territory_fragments.png",
    },
    "provinces": {
        "image": "province_image.png",
        "metadata": "province_data.json",
        "output": "province_fragments.png",
    },
}


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Invalid hex color: {color}")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def to_grayscale_value(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return int(round(0.299 * r + 0.587 * g + 0.114 * b))


def highlight_color(region_index: int, fragment_index: int) -> tuple[int, int, int]:
    key = (region_index + 1) * 92821 + (fragment_index + 1) * 68917
    r = (key * 73 + 17) % 256
    g = (key * 151 + 59) % 256
    b = (key * 199 + 101) % 256
    return int(r), int(g), int(b)


def resolve_seed(region: dict, width: int, height: int) -> tuple[int, int] | None:
    seed = region.get("seed")
    if isinstance(seed, (list, tuple)) and len(seed) == 2:
        sx = int(round(float(seed[0])))
        sy = int(round(float(seed[1])))
    elif "global_x" in region and "global_y" in region:
        sx = int(round(float(region["global_x"])))
        sy = int(round(float(region["global_y"])))
    elif "local_x" in region and "local_y" in region:
        sx = int(round(float(region["local_x"])))
        sy = int(round(float(region["local_y"])))
    else:
        return None

    sx = max(0, min(width - 1, sx))
    sy = max(0, min(height - 1, sy))
    return sx, sy


def choose_seed_component(
    labels: np.ndarray,
    region_mask: np.ndarray,
    seed_xy: tuple[int, int],
) -> int:
    sx, sy = seed_xy
    h, w = labels.shape
    sx = max(0, min(w - 1, int(sx)))
    sy = max(0, min(h - 1, int(sy)))
    comp = int(labels[sy, sx])
    if comp > 0:
        return comp

    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        return 0

    dx = xs.astype(np.float64) - float(sx)
    dy = ys.astype(np.float64) - float(sy)
    nearest_idx = int(np.argmin(dx * dx + dy * dy))
    return int(labels[int(ys[nearest_idx]), int(xs[nearest_idx])])


def resolve_bbox(region: dict, width: int, height: int) -> tuple[int, int, int, int]:
    bbox = region.get("bbox") or region.get("bbox_local")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0, 0, width, height

    x0 = int(np.floor(float(bbox[0])))
    y0 = int(np.floor(float(bbox[1])))
    x1 = int(np.ceil(float(bbox[2])))
    y1 = int(np.ceil(float(bbox[3])))

    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(x0, min(width, x1))
    y1 = max(y0, min(height, y1))
    return x0, y0, x1, y1


def detect_fragmented_regions(
    region_map: NDArray[np.uint32],
    metadata: list[dict],
) -> tuple[NDArray[np.uint8], dict[str, int], list[dict[str, Any]]]:
    height, width = region_map.shape[:2]
    invalid_region_id = np.iinfo(np.uint32).max

    output = np.zeros((height, width, 4), dtype=np.uint8)
    output[:, :, 3] = 255

    stats = {
        "regions_checked": 0,
        "regions_fragmented": 0,
        "fragment_components": 0,
        "fragment_pixels": 0,
        "regions_missing_seed": 0,
    }
    fragment_details: list[dict[str, Any]] = []

    for region_index, region in enumerate(metadata):
        color_hex = region.get("color")
        region_rgb: tuple[int, int, int] | None = None
        if isinstance(color_hex, str):
            try:
                region_rgb = hex_to_rgb(color_hex)
            except ValueError:
                region_rgb = None

        x0, y0, x1, y1 = resolve_bbox(region, width, height)
        if x1 <= x0 or y1 <= y0:
            continue

        seed_xy = resolve_seed(region, width, height)
        if seed_xy is None:
            stats["regions_missing_seed"] += 1
            region_id_value = None
        else:
            sx, sy = seed_xy
            seed_region_id = int(region_map[sy, sx])
            region_id_value = None if seed_region_id == int(invalid_region_id) else seed_region_id

        if region_id_value is None:
            region_window_ids = region_map[y0:y1, x0:x1]
            valid_window = region_window_ids != invalid_region_id
            if not np.any(valid_window):
                continue
            candidate_ids = region_window_ids[valid_window]
            unique_ids, counts = np.unique(candidate_ids, return_counts=True)
            region_id_value = int(unique_ids[int(np.argmax(counts))])

        region_window_ids = region_map[y0:y1, x0:x1]
        region_mask = region_window_ids == np.uint32(region_id_value)
        if not np.any(region_mask):
            continue

        stats["regions_checked"] += 1

        labels, num_components = ndimage.label(region_mask, structure=FOUR_CONNECTED)

        window_out = output[y0:y1, x0:x1]
        if region_rgb is not None:
            gray = to_grayscale_value(region_rgb)
            window_out[region_mask] = [gray, gray, gray, 255]
        else:
            window_out[region_mask] = [128, 128, 128, 255]

        if num_components <= 1:
            continue

        if seed_xy is None:
            component_sizes = np.bincount(labels[region_mask])
            if component_sizes.size <= 1:
                continue
            component_sizes[0] = 0
            primary_component = int(np.argmax(component_sizes))
        else:
            sx, sy = seed_xy
            local_seed = (sx - x0, sy - y0)
            primary_component = choose_seed_component(labels, region_mask, local_seed)
            if primary_component <= 0:
                component_sizes = np.bincount(labels[region_mask])
                if component_sizes.size <= 1:
                    continue
                component_sizes[0] = 0
                primary_component = int(np.argmax(component_sizes))

        has_fragments = False
        for component_id in range(1, num_components + 1):
            if component_id == primary_component:
                continue

            fragment_mask = labels == component_id
            if not np.any(fragment_mask):
                continue

            has_fragments = True
            stats["fragment_components"] += 1
            stats["fragment_pixels"] += int(np.count_nonzero(fragment_mask))

            r, g, b = highlight_color(region_index, component_id)
            window_out[fragment_mask] = [r, g, b, 255]
            fragment_details.append(
                {
                    "region_id": region.get("region_id"),
                    "map_region_id": int(region_id_value),
                    "component_id": int(component_id),
                    "pixel_count": int(np.count_nonzero(fragment_mask)),
                    "fragment_color": rgb_to_hex((r, g, b)),
                    "source_region_color": color_hex,
                }
            )

        if has_fragments:
            stats["regions_fragmented"] += 1

    return output, stats, fragment_details
