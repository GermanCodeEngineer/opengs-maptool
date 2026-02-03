import sys
import json
import pathlib
import config
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from logic.export_module import _export_provinces_csv_to, _export_territories_csv_to, _export_territories_json_to
from logic.grid_province_generator import generate_provinces_grid_based
from logic.boundaries_to_cont import convert_boundaries_to_cont_areas, classify_pixels_by_color
from logic.cont_to_territories import convert_cont_areas_to_territories


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
    type_image_path = output_directory / "typeimage.png"
    type_counts_path = output_directory / "typecounts.json"
    territory_viz_image_path = output_directory / "territoryvizimage.png"
    
    
    # Create helper images if necessary
    if cont_areas_image_path.exists():
        cont_areas_image = np.array(Image.open(cont_areas_image_path))
    else:
        cont_areas_image = convert_boundaries_to_cont_areas(np.array(Image.open(boundary_image_path)))
        Image.fromarray(cont_areas_image).save(cont_areas_image_path)

    if type_image_path.exists() and type_counts_path.exists():
        type_image = np.array(Image.open(type_image_path))
        type_counts = json.loads(type_counts_path.read_text())
    else:
        type_image, type_counts = classify_pixels_by_color(np.array(Image.open(land_image_path)), export_colors=True)
        Image.fromarray(type_image).save(type_image_path)
        type_counts_path.write_text(json.dumps(type_counts))

    # Create territory visualization image if necessary
    if territory_viz_image_path.exists():
        viz_image = np.array(Image.open(territory_viz_image_path))
    else:
        viz_image = convert_cont_areas_to_territories(
            cont_areas_image,
            type_image,
            type_counts,
            total_num_territories=1000,
            #ilerp(config.LAND_TERRITORIES_MIN, config.LAND_TERRITORIES_MAX, land_territories_ratio),
        )
        Image.fromarray(viz_image).save(territory_viz_image_path)
        


def main():
    # Default paths
    input_directory = pathlib.Path(__file__).parent / "example_input"
    output_directory = pathlib.Path(__file__).parent / "output"
    
    # Run with default settings
    generate_map(
        input_directory=input_directory,
        output_directory=output_directory,
        boundary_image_name="bound2.png",
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
