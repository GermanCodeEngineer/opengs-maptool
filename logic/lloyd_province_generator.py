import numpy as np
from PIL import Image
import config
from logic.utils import NumberSeries, combine_maps
from logic.lloyd_utils import (
    build_masks,
    generate_jitter_seeds,
    lloyd_relaxation,
    assign_regions,
    assign_borders,
    build_metadata,
)


province_colors: set[tuple[int, int, int]] = set()


def generate_province_map_lloyd_all_colors(
    boundary_image: np.typing.NDArray[np.uint8] | None,
    land_image: np.typing.NDArray[np.uint8],
    points_per_color: int = 10,
    sea_points: int = 5,
    iterations: int = 3,
) -> tuple[Image.Image, list[dict]]:
    """
    Generate province maps for each color in the land image using Lloyd relaxation.
    
    Args:
        boundary_image: Image array with boundaries
        land_image: Image array with distinct colors representing different regions
        points_per_color: Number of provinces to generate per color
        sea_points: Number of sea regions to generate
        iterations: Number of Lloyd relaxation iterations
        
    Returns:
        Combined province image and metadata for all colors
    """
    # Get unique colors from land_image (excluding sea)
    colors = _get_unique_colors(land_image)
    
    all_metadata = []
    h, w = land_image.shape[:2]
    combined_map = np.full((h, w), -1, np.int32)
    
    # Generate provinces for each color
    for color in colors:
        prov_image, metadata = generate_province_map_lloyd_from_images(
            boundary_image=boundary_image,
            land_image=land_image,
            land_points=points_per_color,
            sea_points=0,  # Sea handled separately
            filter_color=color,
            iterations=iterations,
        )
        
        # Merge this color's provinces into combined map
        prov_array = np.array(prov_image)
        mask = (prov_array[:, :, 3] == 255)  # Non-transparent pixels
        combined_map[mask] = prov_array[mask, 0].astype(np.int32)  # Assuming region ID in R channel
        
        all_metadata.extend(metadata)
    
    # Handle sea regions if needed
    if sea_points > 0:
        sea_fill = np.all(land_image == config.OCEAN_COLOR, axis=2)
        sea_border = ~sea_fill
        
        series = NumberSeries(
            config.PROVINCE_ID_PREFIX,
            config.PROVINCE_ID_START,
            config.PROVINCE_ID_END
        )
        
        sea_map, sea_meta, _ = _lloyd_region_map(
            sea_fill, sea_border, sea_points, max(all_metadata) + 1 if all_metadata else 0,
            "ocean", series, province_colors, iterations
        )
        sea_array = np.array(Image.fromarray(sea_map))
        sea_mask = (sea_array[:, :, 3] == 255)
        combined_map[sea_mask] = sea_array[sea_mask, 0].astype(np.int32)
        all_metadata.extend(sea_meta)
    
    # Convert combined map back to image
    province_image = Image.fromarray(combined_map.astype(np.uint8))
    
    return province_image, all_metadata


def _get_unique_colors(
    image: np.typing.NDArray[np.uint8],
) -> list[tuple[int, int, int]]:
    """Extract unique RGB colors from image, excluding the ocean color."""
    if image.shape[2] >= 3:
        # Reshape to list of colors
        colors = image.reshape(-1, image.shape[2])[:, :3]
        unique_colors = np.unique(colors, axis=0)
        
        # Filter out ocean color
        ocean_color = np.array(config.OCEAN_COLOR, dtype=np.uint8)
        unique_colors = [
            tuple(c) for c in unique_colors
            if not np.array_equal(c, ocean_color)
        ]
        
        return unique_colors
    else:
        raise ValueError("Image must have at least 3 channels (RGB)")



def _create_color_mask(
    image: np.typing.NDArray[np.uint8],
    filter_color: tuple[int, int, int] | tuple[int, int, int, int],
) -> np.typing.NDArray[np.bool_]:
    """Create a boolean mask for pixels matching the filter color."""
    if len(image.shape) < 3:
        raise ValueError("Image must have at least 3 channels (RGB or RGBA)")
    
    channels = image.shape[2]
    
    if len(filter_color) == 3 and channels >= 3:
        # RGB color match
        mask = (
            (image[:, :, 0] == filter_color[0]) &
            (image[:, :, 1] == filter_color[1]) &
            (image[:, :, 2] == filter_color[2])
        )
    elif len(filter_color) == 4 and channels >= 4:
        # RGBA color match
        mask = (
            (image[:, :, 0] == filter_color[0]) &
            (image[:, :, 1] == filter_color[1]) &
            (image[:, :, 2] == filter_color[2]) &
            (image[:, :, 3] == filter_color[3])
        )
    else:
        raise ValueError(f"Filter color length ({len(filter_color)}) doesn't match image channels ({channels})")
    
    return mask


def generate_province_map_lloyd_from_images(
    boundary_image: np.typing.NDArray[np.uint8] | None,
    land_image: np.typing.NDArray[np.uint8] | None,
    land_points: int,
    sea_points: int,
    filter_color: tuple[int, int, int] | tuple[int, int, int, int],
    iterations: int = 3,
) -> tuple[Image.Image, list[dict]]:
    """
    Generate province map using Lloyd relaxation.
    
    Args:
        boundary_image: Image array with boundaries
        land_image: Image array indicating land areas
        land_points: Number of land regions to generate
        sea_points: Number of sea regions to generate
        filter_color: Only generate regions from pixels matching this color (RGB or RGBA)
        iterations: Number of Lloyd relaxation iterations
    """
    province_colors.clear()

    land_fill, land_border, sea_fill, sea_border, land_mask, sea_mask = build_masks(
        boundary_image, land_image
    )
    
    # Apply color filter if specified
    if filter_color is not None and land_image is not None:
        color_mask = _create_color_mask(land_image, filter_color)
        land_fill = land_fill & color_mask
        land_mask = land_mask & color_mask

    series = NumberSeries(
        config.PROVINCE_ID_PREFIX,
        config.PROVINCE_ID_START,
        config.PROVINCE_ID_END
    )

    land_map, land_meta, next_index = _lloyd_region_map(
        land_fill, land_border, land_points, 0, "land", series, province_colors, iterations
    )

    if sea_points > 0 and land_image is not None:
        sea_map, sea_meta, _ = _lloyd_region_map(
            sea_fill, sea_border, sea_points, next_index, "ocean", series, province_colors, iterations
        )
    else:
        h, w = land_fill.shape
        sea_map = np.full((h, w), -1, np.int32)
        sea_meta = []

    metadata = land_meta + sea_meta

    province_image = combine_maps(
        land_map, sea_map, metadata, land_mask, sea_mask
    )

    return province_image, metadata


def _lloyd_region_map(
    fill_mask: np.ndarray,
    border_mask: np.ndarray,
    num_points: int,
    start_index: int,
    ptype: str,
    series: NumberSeries,
    used_colors: set[tuple[int, int, int]],
    iterations: int,
) -> tuple[np.ndarray, list[dict], int]:
    if num_points <= 0 or not fill_mask.any():
        empty = np.full(fill_mask.shape, -1, np.int32)
        return empty, [], start_index

    seeds = generate_jitter_seeds(fill_mask, num_points)
    seeds = [(x, y) for x, y in seeds if fill_mask[y, x]]

    if not seeds:
        empty = np.full(fill_mask.shape, -1, np.int32)
        return empty, [], start_index

    seeds = lloyd_relaxation(fill_mask, seeds, iterations, boundary_mask=border_mask)

    pmap = assign_regions(fill_mask, seeds, start_index)
    assign_borders(pmap, border_mask)

    metadata = build_metadata(
        pmap, seeds, start_index, ptype, series, used_colors, is_territory=False
    )

    next_index = start_index + len(metadata)
    return pmap, metadata, next_index
