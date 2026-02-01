import config
import numpy as np
from PIL import Image
from logic.utils import NumberSeries, is_sea_color, combine_maps, create_region_map

territory_colors: set[tuple[int, int, int]] = set()


def generate_territory_map(main_layout):
    territory_colors.clear()
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

    # NUMBER SERIES FOR TERRITORIES
    series = NumberSeries(
        config.TERRITORY_ID_PREFIX,
        config.TERRITORY_ID_START,
        config.TERRITORY_ID_END
    )

    # GENERATE TERRITORIES
    land_points = main_layout.territory_land_slider.value()
    sea_points = main_layout.territory_ocean_slider.value()

    start_index = 0

    land_map, land_meta, next_index = create_region_map(
        land_fill, land_border, land_points, start_index, "land", series, territory_colors, is_territory=True,
    )

    main_layout.progress.setValue(50)

    if sea_points > 0 and land_image is not None:
        sea_map, sea_meta, _ = create_region_map(
            sea_fill, sea_border, sea_points, next_index, "ocean", series, territory_colors, is_territory=True,
        )
    else:
        sea_map = np.full((map_h, map_w), -1, np.int32)
        sea_meta = []

    metadata = land_meta + sea_meta

    # Build raw territory image (not displayed)
    territory_image = combine_maps(
        land_map, sea_map, metadata, land_mask, sea_mask
    )

    # Build lookup from color -> territory_id
    color_to_id = {}
    for d in metadata:
        color_to_id[(d["R"], d["G"], d["B"])] = d["territory_id"]

    # Build territory -> province list
    terrain_province_map = {}

    province_data = main_layout.province_data
    territory_pixels = territory_image.load()

    for province in province_data:
        x = province["x"]
        y = province["y"]

        r, g, b = territory_pixels[x, y]
        tid = color_to_id.get((r, g, b))
        if tid is None:
            continue

        terrain_province_map.setdefault(
            tid, []).append(province["province_id"])

    # Attach province_ids to territory metadata
    for d in metadata:
        tid = d["territory_id"]
        d["province_ids"] = terrain_province_map.get(tid, [])

    # Build province-based territory image
    province_image = main_layout.province_image_display.get_image()
    territory_province_image = build_province_based_territory_image(
        province_image,
        province_data,
        metadata
    )

    # Display THIS instead of the raw territory map
    main_layout.territory_image_display.set_image(territory_province_image)
    main_layout.territory_data = metadata

    main_layout.progress.setValue(100)
    main_layout.terrain_province_map = terrain_province_map

    # print(terrain_province_map)
    main_layout.button_exp_terr_img.setEnabled(True)
    main_layout.button_exp_terr_csv.setEnabled(True)
    main_layout.button_exp_terr_json.setEnabled(True)
    return territory_province_image, metadata


def build_province_based_territory_image(province_image, province_data, territory_data):

    p_arr = np.array(province_image, copy=False)
    h, w, _ = p_arr.shape

    # Build lookup: province_id -> territory color
    province_to_territory_color = {}

    for terr in territory_data:
        tcolor = (terr["R"], terr["G"], terr["B"])
        for pid in terr["province_ids"]:
            province_to_territory_color[pid] = tcolor

    # Build lookup: province color -> province_id
    color_to_pid = {}
    for p in province_data:
        color_to_pid[(p["R"], p["G"], p["B"])] = p["province_id"]

    out = np.zeros((h, w, 3), np.uint8)

    for y in range(h):
        for x in range(w):
            rgb = tuple(p_arr[y, x])
            pid = color_to_pid.get(rgb)
            if pid is None:
                continue

            terr_color = province_to_territory_color.get(pid)
            if terr_color is None:
                continue

            out[y, x] = terr_color

    return Image.fromarray(out)
