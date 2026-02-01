import config
import numpy as np
from logic.utils import NumberSeries, is_sea_color, combine_maps, create_region_map

province_colors: set[tuple[int, int, int]] = set()


def generate_province_map(main_layout):
    province_colors.clear()
    main_layout.progress.setVisible(True)
    main_layout.progress.setValue(10)

    boundary_image = main_layout.boundary_image_display.get_image()
    land_image = main_layout.land_image_display.get_image()

    if boundary_image is None and land_image is None:
        raise ValueError(
            "Need at least boundary OR ocean image to determine map size.")

    # BOUNDARY MASK
    if boundary_image is not None:
        b_arr = np.array(boundary_image, copy=False)

        if b_arr.ndim == 3:
            r, g, b = config.BOUNDARY_COLOR
            boundary_mask = (
                (b_arr[..., 0] == r) &
                (b_arr[..., 1] == g) &
                (b_arr[..., 2] == b)
            )
        else:
            (val,) = config.BOUNDARY_COLOR[:1]
            boundary_mask = (b_arr == val)

        map_h, map_w = boundary_mask.shape

    else:
        boundary_mask = None

    # LAND / SEA MASKS
    if land_image is not None:
        o_arr = np.array(land_image, copy=False)
        sea_mask = is_sea_color(o_arr)
        land_mask = ~sea_mask

        if boundary_mask is None:
            map_h, map_w = sea_mask.shape
    else:
        if boundary_mask is None:
            raise ValueError("Could not determine map size.")

        sea_mask = np.zeros((map_h, map_w), dtype=bool)
        land_mask = np.ones((map_h, map_w), dtype=bool)

    if boundary_mask is None:
        land_fill = land_mask
        land_border = sea_mask

        sea_fill = sea_mask
        sea_border = land_mask
    else:
        land_fill = land_mask & ~boundary_mask
        land_border = boundary_mask | sea_mask

        sea_fill = sea_mask & ~boundary_mask
        sea_border = boundary_mask | land_mask

    # CREATE NUMBER SERIES
    series = NumberSeries(
        config.PROVINCE_ID_PREFIX,
        config.PROVINCE_ID_START,
        config.PROVINCE_ID_END
    )

    # GENERATE PROVINCES
    land_points = main_layout.land_slider.value()
    sea_points = main_layout.ocean_slider.value()

    land_map, land_meta, next_index = create_region_map(
        land_fill, land_border, land_points, 0, "land", series, province_colors, is_territory=False,
    )

    main_layout.progress.setValue(50)

    if sea_points > 0 and land_image is not None:
        sea_map, sea_meta, _ = create_region_map(
            sea_fill, sea_border, sea_points, next_index, "ocean", series, province_colors, is_territory=False,
        )
    else:
        sea_map = np.full((map_h, map_w), -1, np.int32)
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
