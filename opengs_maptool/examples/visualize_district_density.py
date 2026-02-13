"""
Create a district-density visualization image from exported district data.

By default, this script reads:
- examples/output/districts_data.json

And writes:
- examples/output/district_density_visualization.png

Default mode is pixel-accurate (same resolution as districts_image.png).
Use --mode bbox for a fast approximate heatmap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Invalid color: {color}")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


def density_to_rgb(normalized_density: np.ndarray) -> np.ndarray:
    """
    Convert normalized density values (0..1) to RGB colors.

    Color ramp: dark blue -> cyan -> yellow -> red
    """
    stops = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    ramp = np.array(
        [
            [20, 32, 120],
            [38, 196, 236],
            [250, 224, 77],
            [210, 35, 35],
        ],
        dtype=np.float32,
    )

    r = np.interp(normalized_density, stops, ramp[:, 0])
    g = np.interp(normalized_density, stops, ramp[:, 1])
    b = np.interp(normalized_density, stops, ramp[:, 2])

    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def parse_district_data(data_path: Path) -> list[dict]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of districts in {data_path}")
    return data


def make_heatmap_from_image(districts: list[dict], districts_image_path: Path) -> Image.Image:
    source_image = Image.open(districts_image_path).convert("RGB")
    source = np.array(source_image)

    density_by_color: dict[tuple[int, int, int], float] = {}
    densities: list[float] = []

    for district in districts:
        color = district.get("color")
        density = district.get("density_multiplier")
        if color is None or density is None:
            continue
        rgb = hex_to_rgb(str(color))
        density_value = float(density)
        density_by_color[rgb] = density_value
        densities.append(density_value)

    if not densities:
        raise ValueError("No density values found in district data")

    min_density = min(densities)
    max_density = max(densities)
    density_range = max(max_density - min_density, 1e-9)

    packed = (
        (source[:, :, 0].astype(np.uint32) << 16)
        | (source[:, :, 1].astype(np.uint32) << 8)
        | source[:, :, 2].astype(np.uint32)
    )
    flat = packed.reshape(-1)
    unique_colors, inverse_indices = np.unique(flat, return_inverse=True)

    density_lookup: dict[int, float] = {
        (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]: density
        for rgb, density in density_by_color.items()
    }

    unique_densities = np.full(unique_colors.shape[0], np.nan, dtype=np.float32)
    for index, color_key in enumerate(unique_colors):
        density_value = density_lookup.get(int(color_key))
        if density_value is not None:
            unique_densities[index] = density_value

    density_map = unique_densities[inverse_indices].reshape(source.shape[0], source.shape[1])

    valid_mask = ~np.isnan(density_map)
    normalized = np.zeros_like(density_map, dtype=np.float32)
    normalized[valid_mask] = (density_map[valid_mask] - min_density) / density_range

    rgb_heat = np.zeros((source.shape[0], source.shape[1], 3), dtype=np.uint8)
    rgb_heat[valid_mask] = density_to_rgb(normalized[valid_mask])

    alpha = np.zeros((source.shape[0], source.shape[1]), dtype=np.uint8)
    alpha[valid_mask] = 255

    rgba = np.dstack([rgb_heat, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def make_heatmap_from_bboxes(districts: list[dict]) -> Image.Image:
    valid = [d for d in districts if isinstance(d.get("bbox"), list) and len(d["bbox"]) == 4]
    if not valid:
        raise ValueError("No valid bounding boxes found in district data")

    max_x = max(int(d["bbox"][2]) for d in valid)
    max_y = max(int(d["bbox"][3]) for d in valid)

    canvas = Image.new("RGBA", (max_x + 1, max_y + 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    density_values = [float(d.get("density_multiplier", 1.0)) for d in valid]
    min_density = min(density_values)
    max_density = max(density_values)
    density_range = max(max_density - min_density, 1e-9)

    for district in valid:
        x0, y0, x1, y1 = district["bbox"]
        density = float(district.get("density_multiplier", 1.0))
        normalized = (density - min_density) / density_range
        color = density_to_rgb(np.array([normalized], dtype=np.float32))[0]
        draw.rectangle([x0, y0, x1, y1], fill=(int(color[0]), int(color[1]), int(color[2]), 220))

    return canvas


def build_parser() -> argparse.ArgumentParser:
    example_output = Path(__file__).parent / "output"
    parser = argparse.ArgumentParser(description="Visualize district density as an image")
    parser.add_argument(
        "--data",
        type=Path,
        default=example_output / "districts_data.json",
        help="Path to districts_data.json",
    )
    parser.add_argument(
        "--district-image",
        type=Path,
        default=example_output / "districts_image.png",
        help="Path to districts_image.png (used for pixel-accurate heatmap)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=example_output / "district_density_visualization.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "bbox", "pixel"],
        default="auto",
        help="Visualization mode: auto (pixel if image exists), pixel, or bbox",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    districts = parse_district_data(args.data)

    if args.mode == "pixel":
        if not args.district_image.exists():
            raise FileNotFoundError(
                f"Pixel mode requires district image file: {args.district_image}"
            )
        heatmap = make_heatmap_from_image(districts, args.district_image)
        mode = "pixel map"
    elif args.mode == "auto" and args.district_image.exists():
        heatmap = make_heatmap_from_image(districts, args.district_image)
        mode = "pixel map (auto)"
    else:
        heatmap = make_heatmap_from_bboxes(districts)
        mode = "bbox map"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    heatmap.save(args.output)
    print(f"Saved district density visualization ({mode}) to: {args.output}")


if __name__ == "__main__":
    main()
