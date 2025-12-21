import config
import numpy as np
from collections import deque
from PIL import Image
from scipy.ndimage import distance_transform_edt
from logic.numb_gen import NumberSeries

used_colors = set()


def generate_territory_map(main_layout):
    used_colors.clear()
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

    land_map, land_meta, next_index = create_territory_map(
        land_fill, land_border, land_points, start_index, "land", series
    )

    main_layout.progress.setValue(50)

    if sea_points > 0 and land_image is not None:
        sea_map, sea_meta, _ = create_territory_map(
            sea_fill, sea_border, sea_points, next_index, "ocean", series
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


# BASIC UTILITIES

def is_sea_color(arr):
    r, g, b = config.OCEAN_COLOR
    return (arr[..., 0] == r) & (arr[..., 1] == g) & (arr[..., 2] == b)


def _color_from_id(index: int, ptype: str, used_colors=used_colors):
    rng = np.random.default_rng(index + 1)
    while True:
        if ptype == "ocean":
            r = rng.integers(0, 60)
            g = rng.integers(0, 80)
            b = rng.integers(100, 180)
        else:
            r, g, b = map(int, rng.integers(0, 256, 3))

        color = (int(r), int(g), int(b))
        if color not in used_colors:
            used_colors.add(color)
            return color


def generate_jitter_seeds(mask: np.ndarray, num_points: int):
    if num_points <= 0:
        return []

    h, w = mask.shape
    grid = max(1, int(np.sqrt(num_points)))

    cell_h = h / grid
    cell_w = w / grid
    rng = np.random.default_rng()
    seeds = []

    for gy in range(grid):
        y0 = int(gy * cell_h)
        y1 = int((gy + 1) * cell_h)

        for gx in range(grid):
            x0 = int(gx * cell_w)
            x1 = int((gx + 1) * cell_w)

            cell = mask[y0:y1, x0:x1]
            ys, xs = np.where(cell)

            if xs.size == 0:
                continue

            i = rng.integers(xs.size)
            seeds.append((x0 + xs[i], y0 + ys[i]))

    return seeds


def create_territory_map(fill_mask, border_mask, num_points, start_index, ptype, series):
    if num_points <= 0 or not fill_mask.any():
        empty = np.full(fill_mask.shape, -1, np.int32)
        return empty, [], start_index

    seeds = generate_jitter_seeds(fill_mask, num_points)
    seeds = [(x, y) for x, y in seeds if fill_mask[y, x]]

    if not seeds:
        empty = np.full(fill_mask.shape, -1, np.int32)
        return empty, [], start_index

    pmap, metadata = flood_fill(fill_mask, seeds, start_index, ptype, series)
    assign_borders(pmap, border_mask)
    finalize_metadata(metadata)

    next_index = len(metadata)
    return pmap, list(metadata.values()), next_index


def flood_fill(fill_mask, seeds, start_index, ptype, series):
    h, w = fill_mask.shape
    pmap = np.full((h, w), -1, np.int32)

    metadata = {}
    q = deque()

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for i, (sx, sy) in enumerate(seeds):
        index = start_index + i

        tid = series.get_id()
        if tid is None:
            continue

        pmap[sy, sx] = index

        r, g, b = _color_from_id(index, ptype)
        metadata[index] = {
            "territory_id": tid,
            "territory_type": ptype,
            "R": r, "G": g, "B": b,
            "sum_x": sx,
            "sum_y": sy,
            "count": 1
        }

        q.append((sx, sy, index))

    while q:
        x, y, index = q.popleft()
        d = metadata[index]

        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy

            if 0 <= nx < w and 0 <= ny < h:
                if pmap[ny, nx] == -1 and fill_mask[ny, nx]:
                    pmap[ny, nx] = index
                    d["sum_x"] += nx
                    d["sum_y"] += ny
                    d["count"] += 1
                    q.append((nx, ny, index))

    return pmap, metadata


def assign_borders(pmap, border_mask):
    valid = pmap >= 0
    if not valid.any() or not border_mask.any():
        return

    _, (ny, nx) = distance_transform_edt(~valid, return_indices=True)
    bm = border_mask
    pmap[bm] = pmap[ny[bm], nx[bm]]


def finalize_metadata(metadata):
    for d in metadata.values():
        c = d["count"]
        d["x"] = d["sum_x"] / c
        d["y"] = d["sum_y"] / c
        del d["sum_x"], d["sum_y"], d["count"]


def combine_maps(land_map, sea_map, metadata, land_mask, sea_mask):
    if land_map is not None and land_map.size > 0:
        h, w = land_map.shape
    else:
        h, w = sea_map.shape

    combined = np.full((h, w), -1, np.int32)

    if land_map is not None:
        lm = (land_map >= 0) & land_mask
        combined[lm] = land_map[lm]

    if sea_map is not None:
        sm = (sea_map >= 0) & sea_mask
        combined[sm] = sea_map[sm]

    if (combined >= 0).any():
        valid = combined >= 0
        _, (ny, nx) = distance_transform_edt(~valid, return_indices=True)
        missing = combined < 0
        combined[missing] = combined[ny[missing], nx[missing]]

    out = np.zeros((h, w, 3), np.uint8)

    if not metadata:
        return Image.fromarray(out)

    color_lut = np.zeros((len(metadata), 3), np.uint8)

    for index, d in enumerate(metadata):
        color_lut[index] = (d["R"], d["G"], d["B"])

    valid = combined >= 0
    out[valid] = color_lut[combined[valid]]

    return Image.fromarray(out)
