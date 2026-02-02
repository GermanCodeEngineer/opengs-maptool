import config
import numpy as np
from PIL import Image
from collections import deque
from scipy.ndimage import distance_transform_edt

class NumberSeries:
    def __init__(self, PREFIX: str, NUMBER_START: int, NUMBER_END: int) -> None:
        self.PREFIX: str = PREFIX
        self.NUMBER_END: int = NUMBER_END
        self.ID_LENGTH: int = len(str(NUMBER_END))
        self.number_next: int = NUMBER_START

    def get_id(self) -> str:
        if self.number_next > self.NUMBER_END:
            print("ERROR: No more available numbers!")
            return None

        formatted_number: str = self.PREFIX + \
            str(self.number_next).zfill(self.ID_LENGTH)
        self.number_next += 1
        return formatted_number


def color_from_id(index: int, ptype: str, used_colors: set) -> tuple[int, int, int]:
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


def is_sea_color(arr: np.ndarray) -> np.ndarray:
    r, g, b = config.OCEAN_COLOR
    return (arr[..., 0] == r) & (arr[..., 1] == g) & (arr[..., 2] == b)


def generate_jitter_seeds(mask: np.ndarray, num_points: int) -> list[tuple[int, int]]:
    if num_points <= 0:
        return []

    h, w = mask.shape
    grid = max(1, int(np.sqrt(num_points)))

    cell_h = h / grid
    cell_w = w / grid
    rng = np.random.default_rng(config.RNG_SEED)
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


def create_region_map(
        fill_mask: np.ndarray, border_mask: np.ndarray, num_points: int, start_index: int,
        ptype: str, series: NumberSeries, used_colors: set[tuple[int, int, int]], is_territory: bool,
    ) -> tuple[np.ndarray, list, int]:
    if num_points <= 0 or not fill_mask.any():
        empty = np.full(fill_mask.shape, -1, np.int32)
        return empty, [], start_index

    seeds = generate_jitter_seeds(fill_mask, num_points)
    seeds = [(x, y) for x, y in seeds if fill_mask[y, x]]

    if not seeds:
        empty = np.full(fill_mask.shape, -1, np.int32)
        return empty, [], start_index

    pmap, metadata = flood_fill(fill_mask, seeds, start_index, ptype, series, used_colors, is_territory)
    assign_borders(pmap, border_mask)
    finalize_metadata(metadata)

    next_index = len(metadata)
    return pmap, list(metadata.values()), next_index


def flood_fill(fill_mask: np.ndarray, seeds: list[tuple[int, int]], start_index: int, ptype: str, series: NumberSeries, used_colors: set[tuple[int, int, int]], is_territory: bool) -> tuple[np.ndarray, dict]:
    h, w = fill_mask.shape
    pmap = np.full((h, w), -1, np.int32)

    metadata = {}
    q = deque()

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for i, (sx, sy) in enumerate(seeds):
        index = start_index + i

        rid = series.get_id()
        if rid is None:
            continue

        pmap[sy, sx] = index

        r, g, b = color_from_id(index, ptype, used_colors)
        metadata[index] = {
            ("territory_id" if is_territory else "province_id"): rid,
            ("territory_type" if is_territory else "province_type"): ptype,
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


def assign_borders(pmap: np.ndarray, border_mask: np.ndarray) -> None:
    valid = pmap >= 0
    if not valid.any() or not border_mask.any():
        return

    _, (ny, nx) = distance_transform_edt(~valid, return_indices=True)
    bm = border_mask
    pmap[bm] = pmap[ny[bm], nx[bm]]


def finalize_metadata(metadata: dict) -> None:
    for d in metadata.values():
        c = d["count"]
        d["x"] = d["sum_x"] / c
        d["y"] = d["sum_y"] / c
        del d["sum_x"], d["sum_y"], d["count"]


def combine_maps(land_map: np.ndarray | None, sea_map: np.ndarray | None, metadata: list, land_mask: np.ndarray, sea_mask: np.ndarray) -> Image.Image:
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

    # Only fill missing pixels if there are gaps - avoid expensive distance_transform when not needed
    missing = combined < 0
    if missing.any() and (combined >= 0).any():
        # Cache: reuse valid mask to avoid recomputation
        valid = ~missing
        _, (ny, nx) = distance_transform_edt(~valid, return_indices=True)
        combined[missing] = combined[ny[missing], nx[missing]]

    out = np.zeros((h, w, 4), np.uint8)

    if not metadata:
        return Image.fromarray(out, mode="RGBA")

    # Vectorized color LUT creation - faster than loop
    color_lut = np.array([[d["R"], d["G"], d["B"], 255] for d in metadata], dtype=np.uint8)

    valid = combined >= 0
    out[valid] = color_lut[combined[valid]]

    return Image.fromarray(out, mode="RGBA")
