import config
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from logic.utils import color_from_id, NumberSeries


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


def generate_jitter_seeds(mask: np.ndarray, num_points: int) -> list[tuple[int, int]]:
    """Generate jittered seeds that respect boundaries (only placed in mask)."""
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


def lloyd_relaxation(mask: np.ndarray, seeds: list[tuple[int, int]], iterations: int, boundary_mask: np.ndarray | None = None) -> list[tuple[int, int]]:
    if iterations <= 0 or not seeds:
        return seeds

    coords_yx = np.column_stack(np.where(mask))
    if coords_yx.size == 0:
        return seeds

    # Avoid astype - use view or direct construction
    coords_xy = np.flip(coords_yx, axis=1).copy()  # Copy needed for contiguous memory
    if coords_xy.dtype != np.float32:
        coords_xy = coords_xy.astype(np.float32, copy=False)  # Avoid unnecessary copy
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
) -> list[dict]:
    if pmap.size == 0 or not seeds:
        return []

    valid = pmap >= 0
    if not valid.any():
        return []

    indices = pmap[valid] - start_index
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

        r, g, b = color_from_id(start_index + i, ptype, used_colors)
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
