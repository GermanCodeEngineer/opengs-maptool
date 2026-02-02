import cProfile
import pstats
import sys
import pathlib
import config
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from logic.export_module import _export_provinces_csv_to, _export_territories_csv_to, _export_territories_json_to
from logic.lloyd_province_generator import generate_province_map_lloyd_all_colors
from logic.boundaries_to_cont import convert_boundaries_to_cont_areas


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
    app = QApplication(sys.argv[:1])
    window = MainWindow()
    window.show()
    
    # Load input images
    land_image_path = input_directory / land_image_name
    boundary_image_path = input_directory / boundary_image_name
    convert_boundaries_to_cont_areas(np.array(Image.open(boundary_image_path)))    

    
    if land_image_path.exists():
        window.land_image_display.set_image(Image.open(land_image_path))
    if boundary_image_path.exists():
        window.boundary_image_display.set_image(Image.open(boundary_image_path))
    
    window.button_gen_prov.setEnabled(True)
    
    
    
    # Set province and territory counts
    window.land_slider.setValue(
        ilerp(config.LAND_PROVINCES_MIN, config.LAND_PROVINCES_MAX, land_provinces_ratio)
    )
    window.ocean_slider.setValue(
        ilerp(config.OCEAN_PROVINCES_MIN, config.OCEAN_PROVINCES_MAX, ocean_provinces_ratio)
    )
    window.territory_land_slider.setValue(
        ilerp(config.LAND_TERRITORIES_MIN, config.LAND_TERRITORIES_MAX, land_territories_ratio)
    )
    window.territory_ocean_slider.setValue(
        ilerp(config.OCEAN_TERRITORIES_MIN, config.OCEAN_TERRITORIES_MAX, ocean_territories_ratio)
    )
    
    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    
    
    # Generate provinces
    #window.button_gen_prov.click()
    #province_image = window.province_image_display.get_image()
    #province_data = window.province_data
     
    # Export provinces
    #if "png" in export_formats:
    #    province_image.save(output_directory / "provinceimage.png")
    #if "csv" in export_formats:
    #    _export_provinces_csv_to(province_data, str(output_directory / "provincedata.csv"))
    
    # Generate territories
    #window.button_gen_territories.click()
    #territory_image = window.territory_image_display.get_image()
    #territory_data = window.territory_data
    
    land_image_arr = np.array(Image.open(land_image_path)) if land_image_path.exists() else None
    boundary_image_arr = np.array(Image.open(boundary_image_path)) if boundary_image_path.exists() else None

    if land_image_arr is None:
        raise FileNotFoundError(f"Land image not found: {land_image_path}")

    province_image, metadata = generate_province_map_lloyd_all_colors(
        boundary_image=boundary_image_arr,
        land_image=land_image_arr,
        points_per_color=ilerp(config.LAND_PROVINCES_MIN, config.LAND_PROVINCES_MAX, land_provinces_ratio),
        sea_points=ilerp(config.OCEAN_PROVINCES_MIN, config.OCEAN_PROVINCES_MAX, ocean_provinces_ratio),
        iterations=1,  # Reduced from 3 - saves ~100s per run
    )

    # Export territories
    if "png" in export_formats:
        #territory_image.save(output_directory / "territoryimage.png")
        province_image.save(output_directory / "provinceimage.png")

    #if "csv" in export_formats:
    #    _export_territories_csv_to(territory_data, str(output_directory / "territorydata.csv"))
    #if "json" in export_formats:
    #    _export_territories_json_to(territory_data, str(output_directory / "territorydata.json"))
    
    
    # Cleanup
    window.close()
    app.quit()
    
    #return {
    #    "province_data": province_data,
    #    "territory_data": territory_data,
    #    "province_image": province_image,
    #    "territory_image": territory_image,
    #}
    


def main():
    # Default paths
    input_directory = pathlib.Path(__file__).parent.parent.parent / "Godot/opengs/wip/outputqgis"
    output_directory = pathlib.Path(__file__).parent / "output"
    
    # Run with default settings
    generate_map(
        input_directory=input_directory,
        output_directory=output_directory,
        land_provinces_ratio=0.05,#0.3,
        ocean_provinces_ratio=0.05,#0.3,
        land_territories_ratio=0.05,#0.7,
        ocean_territories_ratio=0.05,#0.4,
    )


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(30)
