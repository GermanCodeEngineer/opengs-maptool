"""
Fast grid-based province generator with boundary respect.
Creates naturally similar-sized provinces using adaptive grid subdivision.
"""
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import config
from logic.utils import NumberSeries, combine_maps


def generate_provinces_grid_based(
    boundary_image: np.ndarray | None,
    land_image: np.ndarray,
    num_provinces: int = 100,
    num_sea_provinces: int = 20,
) -> tuple[Image.Image, list[dict]]:
    """
    Generate provinces using grid-based approach (much faster than Lloyd).
    
    Creates evenly-spaced seed points in a grid, then assigns pixels to nearest seed.
    Naturally produces similar-sized provinces.
    
    Args:
        boundary_image: Optional boundary image
        land_image: Land/ocean image (ocean color = (5,20,18))
        num_provinces: Number of land provinces
        num_sea_provinces: Number of ocean provinces
        
    Returns:
        Province image and metadata
    """
    h, w = land_image.shape[:2]
    
    # Identify land and sea regions
    ocean_color = np.array(config.OCEAN_COLOR, dtype=np.uint8)
    is_ocean = np.all(land_image[:, :, :3] == ocean_color, axis=2)
    is_land = ~is_ocean
    
    # Handle boundaries
    if boundary_image is not None:
        boundary_color = np.array(config.BOUNDARY_COLOR[:3], dtype=np.uint8)
        is_boundary = np.all(boundary_image[:, :, :3] == boundary_color, axis=2)
    else:
        is_boundary = np.zeros((h, w), dtype=bool)
    
    # Generate land provinces
    land_seeds = _generate_grid_seeds(is_land & ~is_boundary, num_provinces)
    land_map, land_meta = _assign_to_seeds(
        land_seeds, is_land, is_boundary, h, w, 0, "land", num_provinces
    )
    
    # Generate sea provinces
    sea_seeds = _generate_grid_seeds(is_ocean & ~is_boundary, num_sea_provinces)
    sea_map, sea_meta = _assign_to_seeds(
        sea_seeds, is_ocean, is_boundary, h, w, len(land_meta), "ocean", num_sea_provinces
    )
    
    # Combine maps
    combined = np.full((h, w), -1, np.int32)
    if land_map is not None:
        combined[is_land] = land_map[is_land]
    if sea_map is not None:
        combined[is_ocean] = sea_map[is_ocean]
    
    # Convert to image
    out = np.zeros((h, w, 4), np.uint8)
    all_meta = land_meta + sea_meta
    
    if all_meta:
        color_lut = np.array(
            [[d["R"], d["G"], d["B"], 255] for d in all_meta],
            dtype=np.uint8
        )
        valid = combined >= 0
        out[valid] = color_lut[combined[valid]]
    
    return Image.fromarray(out, mode="RGBA"), all_meta


def _generate_grid_seeds(valid_mask: np.ndarray, num_seeds: int) -> list[tuple[int, int]]:
    """
    Generate evenly-spaced seed points in a grid pattern.
    
    This ensures similar-sized provinces naturally.
    """
    h, w = valid_mask.shape
    
    # Calculate grid dimensions
    grid_cells = int(np.sqrt(num_seeds))
    cell_h = h / grid_cells
    cell_w = w / grid_cells
    
    seeds = []
    for gy in range(grid_cells):
        y_start = int(gy * cell_h)
        y_end = int((gy + 1) * cell_h)
        
        for gx in range(grid_cells):
            x_start = int(gx * cell_w)
            x_end = int((gx + 1) * cell_w)
            
            # Find valid pixels in this grid cell
            cell_region = valid_mask[y_start:y_end, x_start:x_end]
            ys, xs = np.where(cell_region)
            
            if len(xs) > 0:
                # Pick center or random valid pixel in cell
                idx = len(xs) // 2  # Center of valid pixels
                seed_x = x_start + xs[idx]
                seed_y = y_start + ys[idx]
                seeds.append((seed_x, seed_y))
    
    # Pad with additional random seeds if needed
    if len(seeds) < num_seeds:
        ys, xs = np.where(valid_mask)
        if len(xs) > 0:
            rng = np.random.default_rng(config.RNG_SEED)
            needed = num_seeds - len(seeds)
            indices = rng.choice(len(xs), min(needed, len(xs)), replace=False)
            for idx in indices:
                seeds.append((xs[idx], ys[idx]))
    
    return seeds[:num_seeds]


def _assign_to_seeds(
    seeds: list[tuple[int, int]],
    region_mask: np.ndarray,
    boundary_mask: np.ndarray,
    h: int,
    w: int,
    start_index: int,
    region_type: str,
    num_expected: int,
) -> tuple[np.ndarray, list[dict]]:
    """
    Assign pixels to nearest seed using chunked processing (optimized for large images).
    """
    pmap = np.full((h, w), -1, np.int32)
    
    if not seeds or not region_mask.any():
        return pmap, []
    
    seeds_array = np.array(seeds, dtype=np.float32)
    valid_region = region_mask & ~boundary_mask
    
    # Process image in chunks to avoid memory issues and improve cache efficiency
    chunk_size = 256  # Process 256x256 tiles at a time
    
    for y_start in range(0, h, chunk_size):
        y_end = min(y_start + chunk_size, h)
        for x_start in range(0, w, chunk_size):
            x_end = min(x_start + chunk_size, w)
            
            # Extract chunk
            chunk_valid = valid_region[y_start:y_end, x_start:x_end]
            if not chunk_valid.any():
                continue
            
            # Create coordinate grids for this chunk
            chunk_h = y_end - y_start
            chunk_w = x_end - x_start
            yy, xx = np.mgrid[y_start:y_end, x_start:x_end]
            
            # Compute distances from all seeds to all pixels in chunk (vectorized)
            # Shape: (num_seeds, chunk_h, chunk_w)
            dx = xx[None, :, :] - seeds_array[:, 0, None, None]
            dy = yy[None, :, :] - seeds_array[:, 1, None, None]
            distances = dx * dx + dy * dy
            
            # Find nearest seed for each pixel
            nearest_seed_idx = np.argmin(distances, axis=0)
            
            # Assign to pmap only for valid pixels
            pmap[y_start:y_end, x_start:x_end][chunk_valid] = start_index + nearest_seed_idx[chunk_valid]
    
    # Handle boundaries - assign to nearest valid region (optimized)
    if boundary_mask.any():
        # Use a simpler dilation approach instead of distance_transform_edt
        from scipy.ndimage import binary_dilation
        
        boundary_pixels = boundary_mask & (pmap == -1)
        if boundary_pixels.any():
            # Dilate valid regions to fill boundaries (much faster than distance transform)
            temp = pmap.copy()
            for _ in range(10):  # Usually boundaries are narrow
                valid = temp >= 0
                dilated = binary_dilation(valid)
                newly_filled = dilated & ~valid & boundary_pixels
                if not newly_filled.any():
                    break
                
                # Assign from neighbors
                from scipy.ndimage import maximum_filter
                temp[newly_filled] = maximum_filter(temp, size=3)[newly_filled]
            
            pmap[boundary_pixels] = temp[boundary_pixels]
    
    # Build metadata
    metadata = []
    used_colors = set()
    series = NumberSeries(
        config.PROVINCE_ID_PREFIX,
        config.PROVINCE_ID_START,
        config.PROVINCE_ID_END,
    )
    
    # Compute all centroids at once (MUCH faster)
    centroids = {}
    ys_all, xs_all = np.where(pmap >= 0)
    labels_all = pmap[ys_all, xs_all]
    
    for province_idx in range(start_index, start_index + len(seeds)):
        mask = labels_all == province_idx
        if mask.any():
            xs_prov = xs_all[mask]
            ys_prov = ys_all[mask]
            centroids[province_idx] = (float(np.mean(xs_prov)), float(np.mean(ys_prov)))
    
    for i, seed in enumerate(seeds):
        rid = series.get_id()
        if rid is None:
            continue
        
        from logic.utils import color_from_id
        r, g, b = color_from_id(start_index + i, region_type, used_colors)
        
        # Get pre-computed centroid
        province_idx = start_index + i
        if province_idx in centroids:
            cx, cy = centroids[province_idx]
        else:
            cx, cy = float(seed[0]), float(seed[1])
        
        metadata.append({
            "province_id": rid,
            "R": int(r),
            "G": int(g),
            "B": int(b),
            "seed": seed,
            "x": cx,
            "y": cy,
            "type": region_type,
        })
    
    return pmap, metadata
