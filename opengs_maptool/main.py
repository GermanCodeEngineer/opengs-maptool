import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt6.QtWidgets import QApplication
import sys

from . import StepMapTool, ProcessMapTool, MapToolWindow, config, export_to_json, export_to_csv, import_from_json


def main_automatic() -> None:
    # Default paths
    input_directory = Path(__file__).parent / "examples" / "input"
    output_directory = Path(__file__).parent / "examples" / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Use ProcessMapTool for full pipeline
    maptool = ProcessMapTool(
        class_image=Image.open(input_directory / "class2_clean.png"),
        boundary_image=Image.open(input_directory / "bound2_edited.png"),
    )
    result = maptool.generate()
    result.cont_area_image.save(output_directory / "cont_area_image.png")
    result.dens_samp_image.save(output_directory / "dens_samp_image.png")
    result.territory_image.save(output_directory / "territory_image.png")
    result.province_image.save(output_directory / "province_image.png")
    export_to_json(dict(
        cont_areas=result.cont_area_data,
        dens_samps=result.dens_samp_data,
        territories=result.territory_data,
        provinces=result.province_data,
    ), output_directory / "data.json")

def main_gui() -> None:
    app = QApplication(sys.argv)
    # Use MapToolWindow for GUI
    window = MapToolWindow()
    window.show()
    sys.exit(app.exec())



def main_stepwise_export():
    """
    New entry point: runs the full StepMapTool pipeline, exporting after each step to separate files.
    """
    input_directory = Path(__file__).parent / "examples" / "input"
    class_image_file = input_directory    / "class3.png" #"class2_clean.png"
    boundary_image_file = input_directory / "bound3.png" #"bound2_edited.png"
    output_directory = Path(__file__).parent / "examples" / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    def export_formats(data, path: Path):
        export_to_json(data, path)
        export_to_csv(data, str(path).removesuffix(".json") + ".csv")

    
    class_image_buffer = np.array(StepMapTool.clean_class_image(Image.open(class_image_file)))
    boundary_image_buffer = np.array(Image.open(boundary_image_file).convert("RGBA"))

    regenerate = getattr(sys.modules["__main__"], "args", None)
    regenerate = getattr(regenerate, "regenerate", False)

    # Step 1: cont_areas
    cont_area_img_path = output_directory / "cont_area_image.png"
    cont_area_data_path = output_directory / "cont_area_data.json"
    if regenerate or not (cont_area_img_path.exists() and cont_area_data_path.exists()):
        cont_area_image_buffer, cont_area_data = StepMapTool.generate_cont_areas(
            class_image_buffer, boundary_image_buffer,
        )
        Image.fromarray(cont_area_image_buffer).save(cont_area_img_path)
        export_formats(cont_area_data, cont_area_data_path)
    else:
        cont_area_image_buffer = np.array(Image.open(cont_area_img_path).convert("RGBA"))
        cont_area_data = import_from_json(cont_area_data_path)

    # Step 2: dens_samps
    dens_samp_img_path = output_directory / "dens_samp_image.png"
    dens_samp_data_path = output_directory / "dens_samp_data.json"
    if regenerate or not (dens_samp_img_path.exists() and dens_samp_data_path.exists()):
        dens_samp_image_buffer, dens_samp_data = StepMapTool.generate_dens_samps(
            boundary_image_buffer,
            cont_area_image_buffer,
            cont_area_data,
            pixels_per_land_dens_samp=config.PIXELS_PER_LAND_DENS_SAMP_DEFAULT,
            pixels_per_water_dens_samp=config.PIXELS_PER_WATER_DENS_SAMP_DEFAULT,
        )
        Image.fromarray(dens_samp_image_buffer).save(dens_samp_img_path)
        export_formats(dens_samp_data, dens_samp_data_path)
    else:
        dens_samp_image_buffer = np.array(Image.open(dens_samp_img_path).convert("RGBA"))
        dens_samp_data = import_from_json(dens_samp_data_path)

    # Step 3: territories
    territory_img_path = output_directory / "territory_image.png"
    territory_data_path = output_directory / "territory_data.json"
    if regenerate or not (territory_img_path.exists() and territory_data_path.exists()):
        territory_image_buffer, territory_data = StepMapTool.generate_territories(
            boundary_image_buffer,
            dens_samp_image_buffer,
            dens_samp_data,
            pixels_per_land_territory=config.PIXELS_PER_LAND_TERRITORY_DEFAULT,
            pixels_per_water_territory=config.PIXELS_PER_WATER_TERRITORY_DEFAULT,
        )
        Image.fromarray(territory_image_buffer).save(territory_img_path)
        export_formats(territory_data, territory_data_path)
    else:
        territory_image_buffer = np.array(Image.open(territory_img_path).convert("RGBA"))
        territory_data = import_from_json(territory_data_path)

    # Step 4: provinces
    province_img_path = output_directory / "province_image.png"
    province_data_path = output_directory / "province_data.json"
    if regenerate or not (province_img_path.exists() and province_data_path.exists()):
        province_image_buffer, province_data = StepMapTool.generate_provinces(
            boundary_image_buffer,
            territory_image_buffer,
            territory_data,
            pixels_per_land_province=config.PIXELS_PER_LAND_PROVINCE_DEFAULT,
            pixels_per_water_province=config.PIXELS_PER_WATER_PROVINCE_DEFAULT,
        )
        Image.fromarray(province_image_buffer).save(province_img_path)
        export_formats(province_data, province_data_path)
    else:
        province_image_buffer = np.array(Image.open(province_img_path).convert("RGBA"))
        province_data = import_from_json(province_data_path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenGS MapTool entrypoints")
    parser.add_argument("-gui", action="store_true", help="Launch the GUI")
    parser.add_argument("-stepwise-export", action="store_true", help="Run the full stepwise pipeline and export after each step")
    parser.add_argument("-regenerate", action="store_true", help="Regenerate all outputs even if files exist")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.gui:
        main_gui()
    elif hasattr(args, "stepwise_export") and args.stepwise_export:
        main_stepwise_export()
    else:
        main_automatic()
