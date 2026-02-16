import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from PyQt6.QtWidgets import QApplication
from . import MapToolWindow, MapTool


def main_automatic() -> None:
    # Default paths
    input_directory = Path(__file__).parent / "examples" / "input"
    output_directory = Path(__file__).parent / "examples" / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    class DistrictMapTool(MapTool):
        def on_cont_areas_generated(self, cont_areas_image, cont_areas_image_buffer, cont_areas_data):
            cont_areas_image.save(output_directory / "cont_areas_image.png")
            (output_directory / "cont_areas_data.json").write_text(json.dumps(cont_areas_data))
            #input("Continue?")

        def on_districts_generated(self, districts_image, districts_image_buffer, districts_data):
            districts_image.save(output_directory / "district_image.png")
            (output_directory / "district_data.json").write_text(json.dumps(districts_data))
            #input("Continue?")
        
        def on_territories_generated(self, territory_image, territory_image_buffer, territory_data):
            territory_image.save(output_directory / "territory_image.png")
            (output_directory / "territory_data.json").write_text(json.dumps(territory_data))
            #input("Continue?")
        
        def on_provinces_generated(self, province_image, province_image_buffer, province_data):
            province_image.save(output_directory / "province_image.png")
            (output_directory / "province_data.json").write_text(json.dumps(province_data))
            
            
    maptool = DistrictMapTool(
        land_image=Image.open(input_directory / "land2.png"),
        boundary_image=Image.open(input_directory / "bound2_edited.png"),
    )

    result = maptool.generate()
    result.cont_areas_image.save(output_directory / "cont_areas_image.png")
    result.class_image.save(output_directory / "class_image.png")
    result.district_image.save(output_directory / "district_image.png")
    result.territory_image.save(output_directory / "territory_image.png")
    result.province_image.save(output_directory / "province_image.png")
    (output_directory / "data.json").write_text(json.dumps(dict(
        cont_areas=result.cont_areas_data,
        class_counts=result.class_counts,
        districts=result.district_data,
        territories=result.territory_data,
        provinces=result.province_data,
    )))

def main_districts_from_areas(regenerate_areas: bool = False) -> None:
    input_directory = Path(__file__).parent / "examples" / "input"
    output_directory = Path(__file__).parent / "examples" / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    cont_areas_image_path = output_directory / "cont_areas_image.png"
    cont_areas_data_path = output_directory / "cont_areas_data.json"

    maptool = MapTool(
        land_image=Image.open(input_directory / "land2.png"),
        boundary_image=Image.open(input_directory / "bound2_edited.png"),
    )

    if regenerate_areas:
        cont_areas_image, cont_areas_image_buffer, cont_areas_data = maptool._generate_cont_areas()
        cont_areas_image.save(cont_areas_image_path)
        cont_areas_data_path.write_text(json.dumps(cont_areas_data))
    else:
        if not cont_areas_image_path.exists() or not cont_areas_data_path.exists():
            raise FileNotFoundError(
                "Missing precomputed area files. Expected: "
                f"{cont_areas_image_path} and {cont_areas_data_path}"
            )

        cont_areas_image_buffer = np.array(Image.open(cont_areas_image_path).convert("RGBA"), dtype=np.uint8)
        cont_areas_data = json.loads(cont_areas_data_path.read_text())

    class_image, class_image_buffer, class_counts = maptool._generate_type_classification()
    district_image, _, district_data = maptool._generate_districts(
        cont_areas_image=cont_areas_image_buffer,
        cont_areas_data=cont_areas_data,
        class_image=class_image_buffer,
        class_counts=class_counts,
    )

    class_image.save(output_directory / "class_image.png")
    district_image.save(output_directory / "district_image.png")
    (output_directory / "district_data.json").write_text(json.dumps(district_data))

def main_gui() -> None:
    app = QApplication(sys.argv)
    window = MapToolWindow()
    window.show()
    sys.exit(app.exec())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenGS MapTool entrypoints")
    parser.add_argument("-gui", action="store_true", help="Launch the GUI")
    parser.add_argument("-test-districts", action="store_true", help="Generate districts from precomputed areas")
    parser.add_argument(
        "-regenerate-areas",
        action="store_true",
        help="With -test-districts: regenerate continuous areas before district generation",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.gui:
        main_gui()
    elif args.test_districts:
        main_districts_from_areas(regenerate_areas=args.regenerate_areas)
    else:
        main_automatic()
