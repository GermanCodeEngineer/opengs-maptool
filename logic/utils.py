import config
import numpy as np
from numpy.typing import NDArray
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


def poisson_disk_samples(
    mask: NDArray[np.bool],
    num_points: int,
    min_dist: float | None = None,
    k: int = 30,
    border_margin: float = 0.0,
) -> list[tuple[int, int]]:
    """
    Generate roughly evenly spaced points within a mask using Bridson's Poisson-disk sampling.

    Args:
        mask: 2D boolean mask of valid area.
        num_points: Target number of points.
        min_dist: Minimum distance between points. If None, estimated from mask area.
        k: Attempts per active point.
        border_margin: Optional distance (in pixels) to keep away from mask edges.

    Returns:
        List of (x, y) points.
    """
    if num_points <= 0:
        return []

    if mask is None or not mask.any():
        return []

    mask = mask.astype(bool)

    if border_margin > 0:
        dist = distance_transform_edt(mask)
        mask = mask & (dist >= border_margin)
        if not mask.any():
            return []

    area = int(mask.sum())
    if area == 0:
        return []

    if min_dist is None:
        min_dist = np.sqrt(area / (num_points * np.pi))
    r = float(max(1.0, min_dist))
    r2 = r * r

    h, w = mask.shape
    cell_size = r / np.sqrt(2)
    grid_w = int(np.ceil(w / cell_size))
    grid_h = int(np.ceil(h / cell_size))
    grid = -np.ones((grid_h, grid_w), dtype=np.int32)

    rng = np.random.default_rng(config.RNG_SEED)

    ys, xs = np.where(mask)
    if xs.size == 0:
        return []

    first_idx = rng.integers(xs.size)
    first = (int(xs[first_idx]), int(ys[first_idx]))

    samples: list[tuple[int, int]] = [first]
    active: list[int] = [0]

    gx0 = int(first[0] / cell_size)
    gy0 = int(first[1] / cell_size)
    grid[gy0, gx0] = 0

    while active and len(samples) < num_points:
        active_i = int(rng.integers(len(active)))
        s_idx = active[active_i]
        sx, sy = samples[s_idx]
        found = False

        for _ in range(k):
            radius = rng.uniform(r, 2 * r)
            angle = rng.uniform(0.0, 2 * np.pi)
            x = int(round(sx + radius * np.cos(angle)))
            y = int(round(sy + radius * np.sin(angle)))

            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            if not mask[y, x]:
                continue

            gx = int(x / cell_size)
            gy = int(y / cell_size)

            y0 = max(0, gy - 2)
            y1 = min(grid_h, gy + 3)
            x0 = max(0, gx - 2)
            x1 = min(grid_w, gx + 3)

            ok = True
            for ny in range(y0, y1):
                for nx in range(x0, x1):
                    s2_idx = grid[ny, nx]
                    if s2_idx != -1:
                        px, py = samples[s2_idx]
                        if (px - x) ** 2 + (py - y) ** 2 < r2:
                            ok = False
                            break
                if not ok:
                    break

            if ok:
                samples.append((x, y))
                grid[gy, gx] = len(samples) - 1
                active.append(len(samples) - 1)
                found = True
                break

        if not found:
            active.pop(active_i)

    return samples


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
