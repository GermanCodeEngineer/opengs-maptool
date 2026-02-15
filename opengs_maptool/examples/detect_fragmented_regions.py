"""
Detect fragmented regions from a region image + metadata.

Output image convention:
- Non-fragmented pixels are rendered in grayscale.
- Fragment pixels are rendered in colorful highlight colors.

The script uses the metadata `seed` field to decide which connected component
is the primary component of a region. Any other component of the same region
is treated as a fragment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
    image_rgba: np.ndarray,
    metadata: list[dict],
) -> tuple[np.ndarray, dict[str, int], list[dict[str, object]]]:
    height, width = image_rgba.shape[:2]
    rgb_image = image_rgba[:, :, :3]

    output = np.zeros((height, width, 4), dtype=np.uint8)
    output[:, :, 3] = 255

    stats = {
        "regions_checked": 0,
        "regions_fragmented": 0,
        "fragment_components": 0,
        "fragment_pixels": 0,
        "regions_missing_seed": 0,
    }
    fragment_details: list[dict[str, object]] = []

    for region_index, region in enumerate(metadata):
        color_hex = region.get("color")
        if not isinstance(color_hex, str):
            continue

        try:
            region_rgb = hex_to_rgb(color_hex)
        except ValueError:
            continue

        x0, y0, x1, y1 = resolve_bbox(region, width, height)
        if x1 <= x0 or y1 <= y0:
            continue

        rgb_window = rgb_image[y0:y1, x0:x1]
        region_mask = np.all(rgb_window == np.array(region_rgb, dtype=np.uint8), axis=2)
        if not np.any(region_mask):
            continue

        stats["regions_checked"] += 1

        labels, num_components = ndimage.label(region_mask, structure=FOUR_CONNECTED)

        gray = to_grayscale_value(region_rgb)
        window_out = output[y0:y1, x0:x1]
        window_out[region_mask] = [gray, gray, gray, 255]

        if num_components <= 1:
            continue

        seed_xy = resolve_seed(region, width, height)
        if seed_xy is None:
            stats["regions_missing_seed"] += 1
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
                    "component_id": int(component_id),
                    "pixel_count": int(np.count_nonzero(fragment_mask)),
                    "fragment_color": rgb_to_hex((r, g, b)),
                    "source_region_color": color_hex,
                }
            )

        if has_fragments:
            stats["regions_fragmented"] += 1

    return output, stats, fragment_details


def main() -> None:
    base_dir = Path(__file__).parent / "output"

    parser = argparse.ArgumentParser(description="Detect fragmented regions from map image and metadata")
    parser.add_argument("--image", type=Path, default=base_dir / "district_image.png")
    parser.add_argument("--metadata", type=Path, default=base_dir / "district_data.json")
    parser.add_argument("--output", type=Path, default=base_dir / "district_fragments.png")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    image_rgba = np.array(Image.open(args.image).convert("RGBA"), dtype=np.uint8)
    metadata_raw = json.loads(args.metadata.read_text(encoding="utf-8"))

    if not isinstance(metadata_raw, list):
        raise ValueError("Metadata JSON must be a list of region dictionaries")

    visualization, stats, fragment_details = detect_fragmented_regions(image_rgba, metadata_raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visualization).save(args.output)

    print(f"Saved: {args.output}")
    print(
        "stats:",
        json.dumps(stats, indent=2),
    )
    print(
        "fragment_colors:",
        json.dumps(fragment_details, indent=2),
    )


if __name__ == "__main__":
    main()
