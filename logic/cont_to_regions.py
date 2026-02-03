import numpy as np
from numpy.typing import NDArray
from typing import Any
from tqdm import tqdm
from PIL import Image
from logic.lloyd_utils import lloyd_relaxation, assign_regions, build_metadata
from logic.utils import NumberSeries, poisson_disk_samples, color_from_id
import config

def convert_all_cont_areas_to_regions(
        cont_areas_image: NDArray[np.uint8],
        cont_areas_metadata: list[dict[str, Any]],
        type_image: NDArray[np.uint8],
        type_counts: dict[str, int],
        total_num_regions: int,
    ) -> tuple[NDArray[np.uint8], list[dict[str, Any]]]:
    """
    Convert all continuous areas into regions and combine into a single image.
    
    Args:
        cont_areas_image: Continuous areas image (from convert_boundaries_to_cont_areas)
        cont_areas_metadata: Metadata list with area colors (from convert_boundaries_to_cont_areas)
        type_image: Type/classification image
        type_counts: Pixel counts per type
        total_num_regions: Total regions to generate across all areas
    
    Returns:
        Tuple of (combined_image, combined_metadata) where:
        - combined_image: Full-size region image (same dimensions as input)
        - combined_metadata: Flattened list of all region metadata
    """
    h, w = cont_areas_image.shape[:2]
    combined_image = np.zeros((h, w, 4), dtype=np.uint8)
    combined_metadata = []
    
    for area_meta in tqdm(cont_areas_metadata, desc="Converting areas to regions", unit="area"):
        area_id = area_meta["area_id"]
        filter_color = (area_meta["R"], area_meta["G"], area_meta["B"], 255)
        
        region_image, region_metadata, bbox = convert_cont_area_to_regions(
            cont_areas_image=cont_areas_image,
            type_image=type_image,
            type_counts=type_counts,
            total_num_regions=total_num_regions,
            filter_color=filter_color,
        )
        
        # Paste cropped region image back into full-size image (only non-transparent pixels)
        if bbox is not None and len(region_metadata) > 0:
            y_min, y_max, x_min, x_max = bbox
            # Only copy pixels with alpha > 0 to avoid overwriting with transparency
            alpha_mask = region_image[:, :, 3] > 0
            combined_image[y_min:y_max, x_min:x_max][alpha_mask] = region_image[alpha_mask]
            combined_metadata.extend(region_metadata)
    
    return combined_image, combined_metadata

def convert_cont_area_to_regions(
        cont_areas_image: NDArray[np.uint8],
        type_image: NDArray[np.uint8],
        type_counts: dict[str, int],
        total_num_regions: int,
        filter_color: tuple[int, int, int, int],
    ) -> tuple[NDArray[np.uint8], list[dict[str, Any]], tuple[int, int, int, int] | None]:
    """
    Convert a single continous area(usually a country) into an image of regions.
    
    Args:
        cont_areas_image: Continuous areas image
        type_image: Type/classification image
        type_counts: Pixel counts per type
        total_num_regions: Total regions to generate
        filter_color: RGBA color to filter for (default: Germany blue-green)
    
    Returns:
        Tuple of (region_image, region_metadata, bbox) where:
        - region_image: Cropped region image
        - region_metadata: List of region data
        - bbox: (y_min, y_max, x_min, x_max) for pasting back into full image, or None if empty
    """
    # Mask a single country based on filter color
    exact_color = np.array(filter_color, dtype=np.uint8)
    mask = np.all(cont_areas_image == exact_color, axis=2)

    # Find bounding box and crop to region (MUCH faster Lloyd on small regions)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.zeros((10, 10, 4), dtype=np.uint8), [], None
    
    y_min, y_max = rows.min(), rows.max() + 1
    x_min, x_max = cols.min(), cols.max() + 1
    bbox = (y_min, y_max, x_min, x_max)
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    if True: # temp
        Image.fromarray((cropped_mask * 255).astype(np.uint8)).save("debug_cropped_mask.png")

    num_white_pixels = len(rows)
    num_area_regions = max(1, round((num_white_pixels / type_counts["land"]) * total_num_regions))
    
    # Optimization: if only one region, assign entire area to it
    if num_area_regions == 1:
        series = NumberSeries(config.TERRITORY_ID_PREFIX, config.TERRITORY_ID_START, config.TERRITORY_ID_END)
        used_colors = set()
        
        region_id = series.get_id()
        r, g, b = color_from_id(0, "land", used_colors)
        
        # Compute centroid of entire area
        cx = float(np.mean(cols))
        cy = float(np.mean(rows))
        
        # Adjust centroid to cropped coordinates
        cx_cropped = cx - x_min
        cy_cropped = cy - y_min
        
        metadata = [{
            "region_id": region_id,
            "R": int(r),
            "G": int(g),
            "B": int(b),
            "x": cx_cropped,
            "y": cy_cropped,
            "region_type": "land",
        }]
        
        # Fill entire cropped area with single region color
        h, w = cropped_mask.shape
        cropped_image = np.full((h, w, 4), [r, g, b, 255], dtype=np.uint8)
        
        return cropped_image, metadata, bbox
    
    # Multi-region case: use Poisson + Lloyd
    seeds = poisson_disk_samples(cropped_mask, num_area_regions, min_dist=None, k=30, border_margin=0.0)
    
    if not seeds:
        return np.zeros((10, 10, 4), dtype=np.uint8), [], None
    
    # Use Lloyd relaxation on the seeds for better spacing
    seeds = lloyd_relaxation(
        mask=cropped_mask,
        seeds=seeds,
        iterations=2,
        boundary_mask=None,
        fast_mode=False,
    )
    
    series = NumberSeries(config.TERRITORY_ID_PREFIX, config.TERRITORY_ID_START, config.TERRITORY_ID_END)
    used_colors = set()
    
    pmap = assign_regions(cropped_mask, seeds, start_index=0)
    metadata = build_metadata(pmap, seeds, 0, "land", series, used_colors, is_territory=True)
    
    # Convert province map to colored image
    h, w = cropped_mask.shape
    cropped_image = np.zeros((h, w, 4), dtype=np.uint8)
    
    if metadata:
        color_lut = np.array([[d["R"], d["G"], d["B"], 255] for d in metadata], dtype=np.uint8)
        valid = pmap >= 0
        cropped_image[valid] = color_lut[pmap[valid]]
    
    return cropped_image, metadata, bbox

