import numpy as np
from PIL import Image
import config
from logic.utils import NumberSeries, combine_maps
from logic.lloyd_utils import (
    build_masks,
    generate_jitter_seeds,
    lloyd_relaxation,
    assign_regions,
    assign_borders,
    build_metadata,
)


province_colors: set[tuple[int, int, int]] = set()


def generate_province_map_lloyd(main_layout, iterations: int = 3):
    province_colors.clear()
    main_layout.progress.setVisible(True)
    main_layout.progress.setValue(10)

    boundary_image = main_layout.boundary_image_display.get_image()
    land_image = main_layout.land_image_display.get_image()

    land_fill, land_border, sea_fill, sea_border, land_mask, sea_mask = build_masks(
        boundary_image, land_image
    )

    series = NumberSeries(
        config.PROVINCE_ID_PREFIX,
        config.PROVINCE_ID_START,
        config.PROVINCE_ID_END
    )

    land_points = main_layout.land_slider.value()
    sea_points = main_layout.ocean_slider.value()

    land_map, land_meta, next_index = _lloyd_region_map(
        land_fill, land_border, land_points, 0, "land", series, province_colors, iterations
    )

    main_layout.progress.setValue(50)

    if sea_points > 0 and land_image is not None:
        sea_map, sea_meta, _ = _lloyd_region_map(
            sea_fill, sea_border, sea_points, next_index, "ocean", series, province_colors, iterations
        )
    else:
        h, w = land_fill.shape
        sea_map = np.full((h, w), -1, np.int32)
        sea_meta = []

    metadata = land_meta + sea_meta

    province_image = combine_maps(
        land_map, sea_map, metadata, land_mask, sea_mask
    )

    main_layout.province_image_display.set_image(province_image)
    main_layout.province_data = metadata

    main_layout.progress.setValue(100)
    main_layout.button_exp_prov_img.setEnabled(True)
    main_layout.button_exp_prov_csv.setEnabled(True)
    main_layout.button_gen_territories.setEnabled(True)

    return province_image, metadata


def generate_province_map_lloyd_from_images(
    boundary_image: Image.Image | None,
    land_image: Image.Image | None,
    land_points: int,
    sea_points: int,
    iterations: int = 3,
) -> tuple[Image.Image, list[dict]]:
    province_colors.clear()

    land_fill, land_border, sea_fill, sea_border, land_mask, sea_mask = build_masks(
        boundary_image, land_image
    )

    series = NumberSeries(
        config.PROVINCE_ID_PREFIX,
        config.PROVINCE_ID_START,
        config.PROVINCE_ID_END
    )

    land_map, land_meta, next_index = _lloyd_region_map(
        land_fill, land_border, land_points, 0, "land", series, province_colors, iterations
    )

    if sea_points > 0 and land_image is not None:
        sea_map, sea_meta, _ = _lloyd_region_map(
            sea_fill, sea_border, sea_points, next_index, "ocean", series, province_colors, iterations
        )
    else:
        h, w = land_fill.shape
        sea_map = np.full((h, w), -1, np.int32)
        sea_meta = []

    metadata = land_meta + sea_meta

    province_image = combine_maps(
        land_map, sea_map, metadata, land_mask, sea_mask
    )

    return province_image, metadata


def _lloyd_region_map(
    fill_mask: np.ndarray,
    border_mask: np.ndarray,
    num_points: int,
    start_index: int,
    ptype: str,
    series: NumberSeries,
    used_colors: set[tuple[int, int, int]],
    iterations: int,
) -> tuple[np.ndarray, list[dict], int]:
    if num_points <= 0 or not fill_mask.any():
        empty = np.full(fill_mask.shape, -1, np.int32)
        return empty, [], start_index

    seeds = generate_jitter_seeds(fill_mask, num_points)
    seeds = [(x, y) for x, y in seeds if fill_mask[y, x]]

    if not seeds:
        empty = np.full(fill_mask.shape, -1, np.int32)
        return empty, [], start_index

    seeds = lloyd_relaxation(fill_mask, seeds, iterations, boundary_mask=border_mask)

    pmap = assign_regions(fill_mask, seeds, start_index)
    assign_borders(pmap, border_mask)

    metadata = build_metadata(
        pmap, seeds, start_index, ptype, series, used_colors, is_territory=False
    )

    next_index = start_index + len(metadata)
    return pmap, metadata, next_index
