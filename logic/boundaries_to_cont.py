import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from tqdm import tqdm
from logic.utils import ColorSeries
import config

NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def classify_pixels_by_color(
    land_image: NDArray[np.uint8],
    export_colors: bool = False,
) -> tuple[NDArray[np.uint8], dict[str, int]]:
    """
    Classify each pixel as ocean (0), lake (1), or land (2) based on closest color match.
    Return classification image and count.
    
    Args:
        land_image: Input image
        export_colors: If True, return RGBA image with config colors. If False, return classification indices.
    
    Returns:
        If export_colors=False: 2D uint8 array of shape (h, w) with values: 0=ocean, 1=lake, 2=land
        If export_colors=True: RGBA image (h, w, 4) with actual config colors
    """
    
    # Reference colors from config (shape: 3, 3)
    ref_colors = np.array([
        config.OCEAN_COLOR[:3],
        config.LAKE_COLOR[:3],
        config.LAND_COLOR[:3]
    ], dtype=np.float32)
    
    # Get pixel colors (h, w, 3)
    pixels = land_image[:, :, :3].astype(np.float32)
    
    # Compute squared distances to all 3 colors at once (vectorized)
    diff = pixels[:, :, None, :] - ref_colors[None, None, :, :]
    distances_sq = np.sum(diff ** 2, axis=3)  # (h, w, 3)
    
    # Find index of minimum distance (0=ocean, 1=lake, 2=land)
    classification = np.argmin(distances_sq, axis=2).astype(np.uint8)
    
    result: NDArray[np.uint8]
    if export_colors:
        # Create color image using classification as lookup
        color_lut = np.array(
            [
                [*config.OCEAN_COLOR[:3], 255],
                [*config.LAKE_COLOR[:3], 255],
                [*config.LAND_COLOR[:3], 255],
            ],
            dtype=np.uint8,
        )
        result = color_lut[classification]
    else:
        result = classification

    counts_raw = np.bincount(classification.ravel(), minlength=3)
    counts = {
        "ocean": int(counts_raw[0]),
        "lake": int(counts_raw[1]),
        "land": int(counts_raw[2]),
    }
    return result, counts

def convert_boundaries_to_cont_areas(boundaries_image: NDArray[np.uint8], rng_seed: int, min_area_pixels: int = 50) -> tuple[NDArray[np.uint8], list[dict]]:
    """
    Convert the boundary image into an image of continous areas(usually countries).
    
    Args:
        boundaries_image: Input boundary image
        rng_seed: Random seed for color generation
        min_area_pixels: Minimum pixel count for a continuous area to be kept (smaller areas are merged into background)
    
    Returns:
        Tuple of (cont_areas_image, metadata) where metadata contains:
        - region_id: Region ID (1-indexed)
        - R, G, B: Area color
    """

    # Vectorized mask creation - use only R and G channels for border/area detection
    # Area detection: R and G channels are bright (ignore B channel)
    is_white: np.ndarray = (
        (boundaries_image[:, :, 3] == 255) &
        (boundaries_image[:, :, 0] > 180) &
        (boundaries_image[:, :, 1] > 180)
    )

    # Use scipy's label function for connected component analysis (rel. fast)
    white_mask = is_white.astype(np.uint8)
    labeled_array, num_features = ndimage.label(white_mask)

    # Filter out small areas
    if min_area_pixels > 0:
        region_sizes = np.bincount(labeled_array.ravel())
        small_regions = np.where(region_sizes < min_area_pixels)[0]
        for small_region in small_regions:
            if small_region > 0:  # Skip background (0)
                labeled_array[labeled_array == small_region] = 0
        
        # Relabel to have consecutive IDs
        unique_labels = np.unique(labeled_array[labeled_array > 0])
        new_labeled_array = np.zeros_like(labeled_array)
        for new_id, old_id in enumerate(unique_labels, start=1):
            new_labeled_array[labeled_array == old_id] = new_id
        labeled_array = new_labeled_array
        num_features = len(unique_labels)

    # Create image from regions using the labeled array
    regions_image = np.full((*boundaries_image.shape[:2], 4), [0, 0, 0, 255], dtype=np.uint8)
    color_series = ColorSeries(rng_seed, exclude_values=[(0, 0, 0)])
    region_to_color = {}
    metadata = []
    
    # Vectorized color assignment
    for region_id in tqdm(range(1, num_features + 1), desc="Processing boundaries into areas", unit="areas"):
        color_rgb, color_hex = color_series.get_color_rgb_hex(is_water=False)
        region_to_color[region_id] = (*color_rgb, 255)
        regions_image[labeled_array == region_id] = region_to_color[region_id]
        
        # Calculate density multiplier from B channel (0-255)
        # Sample B value from representative points in the region (center and nearby)
        region_mask = labeled_array == region_id
        rows, cols = np.where(region_mask)
        
        if len(rows) > 0:
            # Get center and nearby sample points
            center_row, center_col = int(np.median(rows)), int(np.median(cols))
            sample_points = [
                (center_row, center_col),
                (center_row, max(0, center_col - 5)),
                (center_row, min(boundaries_image.shape[1] - 1, center_col + 5)),
                (max(0, center_row - 5), center_col),
                (min(boundaries_image.shape[0] - 1, center_row + 5), center_col),
            ]
            
            # Sample B channel from these points (avoid out of bounds)
            blue_samples = []
            for r, c in sample_points:
                if 0 <= r < boundaries_image.shape[0] and 0 <= c < boundaries_image.shape[1]:
                    blue_samples.append(boundaries_image[r, c, 2])
            
            avg_blue = float(np.mean(blue_samples)) if blue_samples else 128.0
        else:
            avg_blue = 128.0
        
        # Convert B channel to density multiplier (piecewise linear)
        # B=0 (low blue) -> 0.25 (4x fewer regions)
        # B=128 (mid blue) -> 1.0 (normal)
        # B=255 (high blue) -> 4.0 (4x more regions)
        if avg_blue <= 128.0:
            density_multiplier = (avg_blue / 128) * 0.75 + 0.25
        else:
            density_multiplier = ((avg_blue - 128) / 127) * 3 + 1
        #density_multiplier = max(0.25, min(4.0, density_multiplier))  # Clamp to reasonable range
        
        # Calculate center of mass (centroid) for local coordinates
        center_x = float(np.mean(cols))
        center_y = float(np.mean(rows))
        
        metadata.append({
            "region_id": region_id,
            "color": color_hex,
            "local_x": center_x,
            "local_y": center_y,
            "density_multiplier": density_multiplier,
        })
    
    return regions_image, metadata

def assign_borders_to_areas(
    regions_image: NDArray[np.uint8],
    max_iters: int = 50,
) -> NDArray[np.uint8]:
    """
    Assign black pixels to neighboring areas by 4-neighbor majority vote.

    Args:
        regions_image: RGBA image where non-black pixels represent area colors.
        max_iters: Max number of propagation iterations.

    Returns:
        Updated RGBA image with black pixels filled when possible.
    """

    result = regions_image.copy()
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]

    black_mask = (alpha > 0) & (rgb == 0).all(axis=2)

    if not np.any(black_mask):
        return result

    color_code = (
        rgb[:, :, 0].astype(np.int32) << 16
    ) | (
        rgb[:, :, 1].astype(np.int32) << 8
    ) | rgb[:, :, 2].astype(np.int32)
    color_code[black_mask] = 0

    for _ in range(max_iters):
        if not np.any(black_mask):
            break

        padded = np.pad(color_code, pad_width=1, mode="constant", constant_values=0)
        n0 = padded[:-2, 1:-1]
        n1 = padded[2:, 1:-1]
        n2 = padded[1:-1, :-2]
        n3 = padded[1:-1, 2:]

        v0 = n0 != 0
        v1 = n1 != 0
        v2 = n2 != 0
        v3 = n3 != 0

        c0 = v0 * (1 + (n0 == n1) + (n0 == n2) + (n0 == n3))
        c1 = v1 * (1 + (n1 == n0) + (n1 == n2) + (n1 == n3))
        c2 = v2 * (1 + (n2 == n0) + (n2 == n1) + (n2 == n3))
        c3 = v3 * (1 + (n3 == n0) + (n3 == n1) + (n3 == n2))

        counts = np.stack([c0, c1, c2, c3], axis=0)
        max_count = counts.max(axis=0)

        update_mask = black_mask & (max_count > 0)
        if not np.any(update_mask):
            break

        idx = counts.argmax(axis=0)
        best = np.where(idx == 0, n0, np.where(idx == 1, n1, np.where(idx == 2, n2, n3)))

        color_code[update_mask] = best[update_mask]
        black_mask = color_code == 0

    result[:, :, 0] = (color_code >> 16) & 255
    result[:, :, 1] = (color_code >> 8) & 255
    result[:, :, 2] = color_code & 255
    result[:, :, 3] = 255

    return result
