import numpy as np
from scipy import ndimage
from logic.utils import color_from_id

NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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
    regions_image = np.zeros_like(boundaries_image)
    colors = set()
    region_to_color = {}
    
    # Vectorized color assignment
    for region_id in range(1, num_features + 1):
        region_to_color[region_id] = (*color_from_id(region_id, "land", colors), 255)
        colors.add(region_to_color[region_id])
        regions_image[labeled_array == region_id] = region_to_color[region_id]
    return regions_image
