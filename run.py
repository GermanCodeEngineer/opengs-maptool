import json
import pathlib
import config
import numpy as np
from pathlib import Path
from PIL import Image
from logic.boundaries_to_cont import normalize_area_density, convert_boundaries_to_cont_areas, assign_borders_to_areas, classify_pixels_by_color
from logic.cont_to_regions import convert_all_cont_areas_to_regions
from logic.utils import NumberSeries


def generate_map(
    input_directory: Path | str,
    output_directory: Path | str,
    land_provinces_ratio: float = 0.3,
    ocean_provinces_ratio: float = 0.3,
    land_territories_ratio: float = 0.7,
    ocean_territories_ratio: float = 0.4,
    land_image_name: str = "landimage.png",
    boundary_image_name: str = "boundaryimage.png",
    export_formats: list[str] = None,
) -> dict:
    """
    Generate province and territory maps from input images.
    
    Args:
        input_directory: Directory containing input images
        output_directory: Directory to save generated maps and data
        land_provinces_ratio: Ratio (0-1) for land province count (interpolated between MIN/MAX)
        ocean_provinces_ratio: Ratio (0-1) for ocean province count (interpolated between MIN/MAX)
        land_territories_ratio: Ratio (0-1) for land territory count (interpolated between MIN/MAX)
        ocean_territories_ratio: Ratio (0-1) for ocean territory count (interpolated between MIN/MAX)
        land_image_name: Filename of land/ocean image in input_directory
        boundary_image_name: Filename of boundary image in input_directory
        export_formats: List of formats to export ("png", "csv", "json"). Default: all
    
    Returns:
        Dictionary containing generated data:
            - province_data: list of province metadata
            - territory_data: list of territory metadata
            - province_image: PIL Image
            - territory_image: PIL Image
    """
    if export_formats is None:
        export_formats = ["png", "csv", "json"]
    
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    
    def ilerp(a: int, b: int, t: float) -> int:
        """Integer linear interpolation"""
        return round(a + t * (b - a))
    
    # Create QApplication and window
    #app = QApplication(sys.argv[:1])
    #window = MainWindow()
    #window.show()
    
    # Load input images
    
    land_image_path = input_directory / land_image_name
    boundary_image_path = input_directory / boundary_image_name

    cont_areas_image_path = output_directory / "contareasimage.png"
    cont_areas_data_path = output_directory / "contareasdata.json"
    type_image_path = output_directory / "typeimage.png"
    type_counts_path = output_directory / "typecounts.json"
    territory_image_path = output_directory / "territoryimage.png"
    territory_data_path = output_directory / "territorydata.json"
    province_image_path = output_directory / "provinceimage.png"
    province_data_path = output_directory / "provincedata.json"
    
    # Create helper images if necessary
    if cont_areas_image_path.exists() and cont_areas_data_path.exists():
        cont_areas_image = np.array(Image.open(cont_areas_image_path).convert("RGBA"))
        cont_areas_data = json.loads(cont_areas_data_path.read_text())
    else:
        boundary_image = np.array(Image.open(boundary_image_path).convert("RGBA"))
        areas_with_borders_image, cont_areas_data = convert_boundaries_to_cont_areas(
            boundary_image, 
            config.CONT_AREAS_RNG_SEED,
            min_area_pixels=50  # Filter out tiny areas & islands
        )
        cont_areas_image = assign_borders_to_areas(areas_with_borders_image)
        Image.fromarray(cont_areas_image).save(cont_areas_image_path)
        cont_areas_data_path.write_text(json.dumps(cont_areas_data))
    
    if type_image_path.exists() and type_counts_path.exists():
        type_image = np.array(Image.open(type_image_path).convert("RGBA"))
        type_counts = json.loads(type_counts_path.read_text())
    else:
        type_image, type_counts = classify_pixels_by_color(np.array(Image.open(land_image_path).convert("RGBA")), export_colors=True)
        Image.fromarray(type_image).save(type_image_path)
        type_counts_path.write_text(json.dumps(type_counts))

    # Create territory visualization image if necessary
    if territory_image_path.exists() and territory_data_path.exists():
        territory_image = np.array(Image.open(territory_image_path).convert("RGBA"))
        territory_data = json.loads(territory_data_path.read_text())
    else:
        #number_superseries = NumberSeries(config.AREA_ID_PREFIX, config.SERIES_ID_START, config.SERIES_ID_END)
        territory_image, territory_data = convert_all_cont_areas_to_regions(
            cont_areas_image=cont_areas_image,
            cont_areas_metadata=cont_areas_data,
            type_image=type_image,
            type_counts=type_counts,
            pixels_per_land_region=4500,
            pixels_per_water_region=50000,  # Much larger ocean territories
            fn_new_number_series=lambda area_meta: NumberSeries(
                f"{area_meta['region_id']}-{config.TERRITORY_ID_PREFIX}", config.SERIES_ID_START, config.SERIES_ID_END
            ),
            rng_seed=config.TERRITORIES_RNG_SEED,
            lloyd_iterations=3,
        )

        # Replace ids with correct format
        number_series = NumberSeries(config.TERRITORY_ID_PREFIX, config.SERIES_ID_START, config.SERIES_ID_END)
        for territory in territory_data:
            territory["region_id"] = number_series.get_id()
        Image.fromarray(territory_image).save(territory_image_path)
        territory_data_path.write_text(json.dumps(territory_data))

    """
    # Create province visualization image if necessary
    if province_image_path.exists() and province_data_path.exists():
        province_image = np.array(Image.open(province_image_path).convert("RGBA"))
        province_data = json.loads(province_data_path.read_text())
    else:
        #number_superseries = NumberSeries(config.TERRITORY_ID_PREFIX, config.SERIES_ID_START, config.SERIES_ID_END)
        province_image, province_data = convert_all_cont_areas_to_regions(
            cont_areas_image=territory_image,
            cont_areas_metadata=territory_data,
            type_image=type_image,
            type_counts=type_counts,
            pixels_per_land_region=type_counts.get("land", 1) / (800*4),
            pixels_per_water_region=type_counts.get("ocean", 1) / (300*4), # lakes are insignificant
            fn_new_number_series=lambda territory_meta: NumberSeries(
                f"{territory_meta['region_id']}-{config.PROVINCE_ID_PREFIX}", config.SERIES_ID_START, config.SERIES_ID_END
            ),
            rng_seed=config.PROVINCES_RNG_SEED,
            lloyd_iterations=4,
        )
        Image.fromarray(province_image).save(province_image_path)
        province_data_path.write_text(json.dumps(province_data))
    """


def main():
    # Default paths
    input_directory = pathlib.Path(__file__).parent / "example_input"
    output_directory = pathlib.Path(__file__).parent / "output"
    
    #boundary_image = np.array(Image.open(input_directory / "bound2_borders.png").convert("RGBA"))
    #boundary_image = normalize_area_density(boundary_image)
    #Image.fromarray(boundary_image).save(input_directory / "bound2_yellow.png")

    # Run with default settings
    generate_map(
        input_directory=input_directory,
        output_directory=output_directory,
        boundary_image_name="bound2_density.png",
        land_image_name="land2.png",
        land_provinces_ratio=0.05,#0.3,
        ocean_provinces_ratio=0.05,#0.3,
        land_territories_ratio=0.05,#0.7,
        ocean_territories_ratio=0.05,#0.4,
    )


if __name__ == "__main__":
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(30)
