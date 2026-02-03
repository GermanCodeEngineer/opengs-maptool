import numpy as np
from numpy.typing import NDArray

def convert_cont_areas_to_territories(
        cont_areas_image: NDArray[np.uint8],
        type_image: NDArray[np.uint8],
        type_counts: dict[str, int],
        total_num_territories: int,
    ) -> NDArray[np.uint8]:
    """
    Convert a single continous area(usually a country) into an image of territories.
    """
    # TODO: add parameter for filter color

    # Mask a single country
    # Temporarily just Germany
    exact_color = np.array([16, 249, 240, 255], dtype=np.uint8)
    mask = np.all(cont_areas_image == exact_color, axis=2)

    # Create visualization: white where True, transparent where False
    viz_image = np.zeros((*mask.shape, 4), dtype=np.uint8)
    viz_image[mask] = [255, 255, 255, 255]  # White opaque
    # False pixels remain [0, 0, 0, 0] = transparent

    # Find bounding box
    rows, cols = np.where(mask)
    if len(rows) > 0:
        y_min, y_max = rows.min(), rows.max() + 1
        x_min, x_max = cols.min(), cols.max() + 1
    cropped_image = viz_image[y_min:y_max, x_min:x_max]
    
    num_white_pixels = len(rows)
    num_area_territories = max(1, round((num_white_pixels / type_counts["land"]) * total_num_territories))



    print("Size", num_white_pixels, "of", type_counts["land"], "Ts", num_area_territories, "of", total_num_territories)

    return cropped_image

