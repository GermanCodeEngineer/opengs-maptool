import config
import numpy as np
from hashlib import sha256
from numpy.typing import NDArray
from typing import Any
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def is_sea_color(arr: np.ndarray) -> np.ndarray:
    # Vectorized comparison - faster than individual channel checks
    ocean_color = np.array(config.OCEAN_COLOR, dtype=np.uint8)
    return np.all(arr[..., :3] == ocean_color, axis=-1)


def build_masks(
    boundary_image: np.typing.NDArray[np.uint8] | None,
    land_image: np.typing.NDArray[np.uint8] | None,
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


def lloyd_relaxation(mask: np.ndarray, seeds: list[tuple[int, int]], iterations: int, boundary_mask: np.ndarray | None = None, fast_mode: bool = False) -> list[tuple[int, int]]:
    """
    Lloyd relaxation with optional fast mode.
    
    Args:
        mask: Valid region mask
        seeds: Initial seed positions
        iterations: Number of relaxation iterations (ignored if fast_mode=True)
        boundary_mask: Optional boundary mask
        fast_mode: If True, skip iteration and use single-pass Voronoi (10-20x faster, slightly lower quality)
    """
    if iterations <= 0 or not seeds:
        return seeds

    coords_yx = np.column_stack(np.where(mask))
    if coords_yx.size == 0:
        return seeds

    # Single-pass fast mode: just assign to nearest seed, no iteration
    if fast_mode:
        coords_xy = np.flip(coords_yx, axis=1).astype(np.float32, copy=False)
        tree = cKDTree(np.array(seeds, dtype=np.float32))
        _, labels = tree.query(coords_xy, k=1)
        
        # Return centroid of each region (one pass)
        rng = np.random.default_rng(config.RNG_SEED)
        counts = np.bincount(labels, minlength=len(seeds))
        sum_x = np.bincount(labels, weights=coords_xy[:, 0], minlength=len(seeds))
        sum_y = np.bincount(labels, weights=coords_xy[:, 1], minlength=len(seeds))
        
        new_seeds = []
        for i in range(len(seeds)):
            if counts[i] > 0:
                cx = int(round(sum_x[i] / counts[i]))
                cy = int(round(sum_y[i] / counts[i]))
                cx = max(0, min(cx, mask.shape[1] - 1))
                cy = max(0, min(cy, mask.shape[0] - 1))
                
                # Try to place in valid area
                if mask[cy, cx]:
                    new_seeds.append((cx, cy))
                else:
                    new_seeds.append(seeds[i])
            else:
                new_seeds.append(seeds[i])
        
        return new_seeds

    # Standard Lloyd relaxation (slower but better quality)
    coords_xy = np.flip(coords_yx, axis=1).copy()
    if coords_xy.dtype != np.float32:
        coords_xy = coords_xy.astype(np.float32, copy=False)
    rng = np.random.default_rng(config.RNG_SEED)

    # Cache distance transform (expensive operation)
    _, (ny, nx) = distance_transform_edt(~mask, return_indices=True)

    seeds_arr = np.array(seeds, dtype=np.float32)

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
    ptype: str,
    series: NumberSeries,
    used_colors: set[tuple[int, int, int]],
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

        r, g, b = color_from_string(rid, ptype, used_colors)
        if counts[i] <= 0:
            sx, sy = seeds[i]
            cx, cy = float(sx), float(sy)
        else:
            cx = float(sum_x[i] / counts[i])
            cy = float(sum_y[i] / counts[i])

        metadata.append({
            ("territory_id" if is_territory else "province_id"): rid,
            ("territory_type" if is_territory else "province_type"): ptype,
            "R": r, "G": g, "B": b,
            "x": cx,
            "y": cy,
        })

    return metadata


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


def _color_from_id(index: int, ptype: str, used_colors: set) -> tuple[int, int, int]:
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


def color_from_string(s: str, ptype: str, used_colors: set) -> tuple[int, int, int]:
    """Generate a color from a string (deterministic)."""
    return _color_from_id(abs(int(sha256(s.encode()).hexdigest(), 16)) % 1000000, ptype, used_colors)


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
