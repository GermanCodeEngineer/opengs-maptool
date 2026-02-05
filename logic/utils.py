import config
import numpy as np
from numpy.typing import NDArray
from typing import Any, Iterable
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert RGB values to hex color string (e.g., '#aabbcc')"""
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string (e.g., '#aabbcc') to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class NumberSeries:
    def __init__(self, prefix: str, number_start: int, number_end: int) -> None:
        self.prefix: str = prefix
        self.number_end: int = number_end
        self.id_length: int = len(str(number_end))
        self.number_next: int = number_start

    def get_id(self) -> str | None:
        if self.number_next > self.number_end:
            print("ERROR: No more available numbers!")
            return None

        formatted_number: str = self.prefix + \
            str(self.number_next).zfill(self.id_length)
        self.number_next += 1
        return formatted_number


class NumberSubSeries:
    """Generates hierarchical IDs like 'parent-prefix-1-sub-prefix-1'"""
    def __init__(self, parent_series: NumberSeries, sub_prefix: str, number_start: int, number_end: int) -> None:
        self.parent_series: NumberSeries = parent_series
        self.parent_id = parent_series.get_id()
        self.sub_prefix: str = sub_prefix
        self.sub_number_end: int = number_end
        self.sub_id_length: int = len(str(number_end))
        self.sub_number_next: int = number_start

    def get_id(self) -> str | None:
        if self.sub_number_next > self.sub_number_end:
            print("ERROR: No more available sub numbers!")
            return None

        formatted_sub_number: str = self.sub_prefix + \
            str(self.sub_number_next).zfill(self.sub_id_length)
        self.sub_number_next += 1
        
        return f"{self.parent_id}-{formatted_sub_number}"


class ColorSeries:
    def __init__(self, rng_seed: int, exclude_values: Iterable[tuple[int, int, int]] | None = None) -> None:
        self.rng = np.random.default_rng(rng_seed)
        self.used_values = set() if exclude_values is None else set(exclude_values) 

    def get_color_rgb(self, is_water: bool) -> tuple[int, int, int]:
        while True:
            if is_water:
                r = self.rng.integers(0, 60)
                g = self.rng.integers(0, 80)
                b = self.rng.integers(100, 180)
            else:
                r, g, b = map(int, self.rng.integers(0, 256, 3))

            color = (int(r), int(g), int(b))
            if color not in self.used_values:
                self.used_values.add(color)
                return color
    
    def get_color_hex(self, is_water: bool) -> str:
        return rgb_to_hex(self.get_color_rgb(is_water=is_water))

    def get_color_rgb_hex(self, is_water: bool) -> tuple[tuple[int, int, int], str]:
        rgb = self.get_color_rgb(is_water=is_water)
        return (rgb, rgb_to_hex(rgb))


def is_sea_color(arr: np.ndarray) -> np.ndarray:
    # Vectorized comparison - faster than individual channel checks
    ocean_color = np.array(config.OCEAN_COLOR, dtype=np.uint8)
    return np.all(arr[..., :3] == ocean_color, axis=-1)


def build_masks(
    boundary_image: NDArray[np.uint8] | None,
    land_image: NDArray[np.uint8] | None,
):
    if boundary_image is None and land_image is None:
        raise ValueError("Need at least boundary OR ocean image to determine map size.")

    # Boundary mask
    if boundary_image is not None:
        if boundary_image.ndim == 3:
            r, g, b = config.BOUNDARY_COLOR
            boundary_mask = (
                (boundary_image[..., 0] == r) &
                (boundary_image[..., 1] == g) &
                (boundary_image[..., 2] == b)
            )
        else:
            (val,) = config.BOUNDARY_COLOR[:1]
            boundary_mask = (boundary_image == val)
        map_h, map_w = boundary_mask.shape
    else:
        boundary_mask = None

    # Land / sea mask
    if land_image is not None:
        sea_mask = is_sea_color(land_image)
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

    return land_fill, land_border, sea_fill, sea_border, land_mask, sea_mask


def lloyd_relaxation(
        mask: np.ndarray, point_seeds: list[tuple[int, int]], 
        rng_seed: int, iterations: int, boundary_mask: np.ndarray | None = None
    ) -> list[tuple[int, int]]:
    """
    Lloyd relaxation with optional fast mode.
    
    Args:
        mask: Valid region mask
        point_seeds: Initial seed positions
        rng_seed: RNG seed for reproducibility.
        iterations: Number of relaxation iterations
        boundary_mask: Optional boundary mask
    """
    if iterations <= 0 or not point_seeds:
        return point_seeds

    coords_yx = np.column_stack(np.where(mask))
    if coords_yx.size == 0:
        return point_seeds

    # Standard Lloyd relaxation (slower but better quality)
    coords_xy = np.flip(coords_yx, axis=1).copy()
    if coords_xy.dtype != np.float32:
        coords_xy = coords_xy.astype(np.float32, copy=False)
    rng = np.random.default_rng(rng_seed)

    # Cache distance transform (expensive operation)
    _, (ny, nx) = distance_transform_edt(~mask, return_indices=True)

    seeds_arr = np.array(point_seeds, dtype=np.float32)

    for _ in range(iterations):
        # Assign each coordinate to nearest seed
        tree = cKDTree(seeds_arr)
        _, labels = tree.query(coords_xy, k=1)

        counts = np.bincount(labels, minlength=len(seeds_arr))
        sum_x = np.bincount(labels, weights=coords_xy[:, 0], minlength=len(seeds_arr))
        sum_y = np.bincount(labels, weights=coords_xy[:, 1], minlength=len(seeds_arr))

        for i in range(len(seeds_arr)):
            if counts[i] <= 0:
                idx = rng.integers(0, coords_xy.shape[0])
                seeds_arr[i] = coords_xy[idx]
                continue

            mx = sum_x[i] / counts[i]
            my = sum_y[i] / counts[i]

            cx = int(round(mx))
            cy = int(round(my))
            cx = max(0, min(cx, mask.shape[1] - 1))
            cy = max(0, min(cy, mask.shape[0] - 1))

            # Enforce: seed must be in mask AND NOT in boundary
            if mask[cy, cx]:
                if boundary_mask is None or not boundary_mask[cy, cx]:
                    seeds_arr[i] = (cx, cy)
                else:
                    # Snap to nearest non-boundary point
                    cy2 = int(ny[cy, cx])
                    cx2 = int(nx[cy, cx])
                    seeds_arr[i] = (cx2, cy2)
            else:
                cy2 = int(ny[cy, cx])
                cx2 = int(nx[cy, cx])
                seeds_arr[i] = (cx2, cy2)

    return [(int(x), int(y)) for x, y in seeds_arr]


def assign_regions(mask: np.ndarray, seeds: list[tuple[int, int]], start_index: int) -> np.ndarray:
    """Assign each pixel in mask to nearest seed (respecting mask boundaries)."""
    h, w = mask.shape
    pmap = np.full((h, w), -1, np.int32)

    if not seeds or not mask.any():
        return pmap

    coords_yx = np.column_stack(np.where(mask))
    coords_xy = np.flip(coords_yx, axis=1).astype(np.float32, copy=False)

    # For each valid pixel, find nearest seed using KDTree (Euclidean distance)
    # This respects mask because we only compute distances for pixels in mask
    tree = cKDTree(np.array(seeds, dtype=np.float32))
    _, labels = tree.query(coords_xy, k=1)

    pmap[coords_yx[:, 0], coords_yx[:, 1]] = start_index + labels
    return pmap


def assign_borders(pmap: np.ndarray, border_mask: np.ndarray) -> None:
    """Assign border pixels to nearest valid region (respects boundary structure)."""
    valid = pmap >= 0
    if not valid.any() or not border_mask.any():
        return

    _, (ny, nx) = distance_transform_edt(~valid, return_indices=True)
    bm = border_mask
    pmap[bm] = pmap[ny[bm], nx[bm]]


def build_metadata(
    pmap: np.ndarray,
    seeds: list[tuple[int, int]],
    start_index: int,
    region_type: str,
    series: NumberSeries,
    color_series: ColorSeries,
    is_territory: bool,
) -> list[dict[str, Any]]:
    if pmap.size == 0 or not seeds:
        return []

    valid = pmap >= 0
    if not valid.any():
        return []

    indices = (pmap[valid] - start_index).astype(np.int32)
    # Ensure no negative indices (should not happen, but safeguard against edge cases)
    if np.any(indices < 0):
        indices = np.maximum(indices, 0)
    
    coords_yx = np.column_stack(np.where(valid))
    coords_xy = np.flip(coords_yx, axis=1).astype(np.float32, copy=False)

    counts = np.bincount(indices, minlength=len(seeds))
    sum_x = np.bincount(indices, weights=coords_xy[:, 0], minlength=len(seeds))
    sum_y = np.bincount(indices, weights=coords_xy[:, 1], minlength=len(seeds))

    metadata = []
    for i in range(len(seeds)):
        rid = series.get_id()
        if rid is None:
            continue

        color_hex = color_series.get_color_hex(is_water=(region_type != "land"))
        if counts[i] <= 0:
            sx, sy = seeds[i]
            cx, cy = float(sx), float(sy)
        else:
            cx = float(sum_x[i] / counts[i])
            cy = float(sum_y[i] / counts[i])

        metadata.append({
            ("territory_id" if is_territory else "province_id"): rid,
            ("territory_type" if is_territory else "province_type"): (region_type),
            "color": color_hex,
            "x": cx,
            "y": cy,
        })

    return metadata


def poisson_disk_samples(
    mask: NDArray[np.bool],
    num_points: int,
    rng_seed: int,
    min_dist: float | None = None,
    k: int = 30,
    border_margin: float = 0.0,
) -> list[tuple[int, int]]:
    """
    Generate roughly evenly spaced points within a mask using Bridson's Poisson-disk sampling.

    Args:
        mask: 2D boolean mask of valid area.
        num_points: Target number of points.
        rng_seed: RNG seed for reproducibility.
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

    rng = np.random.default_rng(rng_seed)

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
