import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from logic.utils import color_from_id
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

def convert_boundaries_to_cont_areas(boundaries_image: np.typing.NDArray[np.uint8]) -> np.typing.NDArray[np.uint8]:
    """
    Convert the boundary image into an image of continous areas(usually countries).
    """

    # Vectorized mask creation
    is_white: np.ndarray = (
        (boundaries_image[:, :, 3] == 255) &
        (boundaries_image[:, :, 0] == boundaries_image[:, :, 1]) &
        (boundaries_image[:, :, 1] == boundaries_image[:, :, 2]) &
        (boundaries_image[:, :, :3].sum(axis=2) > 180 * 3)
    )

    # Use scipy's label function for connected component analysis (rel. fast)
    white_mask = is_white.astype(np.uint8)
    labeled_array, num_features = ndimage.label(white_mask)

    # Create image from regions using the labeled array
    regions_image = np.full((*boundaries_image.shape[:2], 4), [0, 0, 0, 255], dtype=np.uint8)
    colors = {(0, 0, 0)} # Forbid black
    region_to_color = {}
    
    # Vectorized color assignment
    for region_id in range(1, num_features + 1):
        region_to_color[region_id] = (*color_from_id(region_id, "land", colors), 255)
        regions_image[labeled_array == region_id] = region_to_color[region_id]
    return regions_image
