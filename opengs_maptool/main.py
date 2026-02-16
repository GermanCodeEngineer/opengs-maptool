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

    class StepMapTool(MapTool):
        def on_cont_areas_generated(self, cont_area_image, cont_area_image_buffer, cont_area_data):
            cont_area_image.save(output_directory / "cont_area_image.png")
            (output_directory / "cont_area_data.json").write_text(json.dumps(cont_area_data))

        def on_districts_generated(self, district_image, district_image_buffer, district_data):
            district_image.save(output_directory / "district_image.png")
            (output_directory / "district_data.json").write_text(json.dumps(district_data))
        
        def on_territories_generated(self, territory_image, territory_image_buffer, territory_data):
            territory_image.save(output_directory / "territory_image.png")
            (output_directory / "territory_data.json").write_text(json.dumps(territory_data))
        
        def on_provinces_generated(self, province_image, province_image_buffer, province_data):
            province_image.save(output_directory / "province_image.png")
            (output_directory / "province_data.json").write_text(json.dumps(province_data))
            
            
    maptool = StepMapTool(
        land_image=Image.open(input_directory / "land2.png"),
        boundary_image=Image.open(input_directory / "bound2_edited.png"),
    )

    result = maptool.generate()
    result.cont_area_image.save(output_directory / "cont_area_image.png")
    result.class_image.save(output_directory / "class_image.png")
    result.district_image.save(output_directory / "district_image.png")
    result.territory_image.save(output_directory / "territory_image.png")
    result.province_image.save(output_directory / "province_image.png")
    (output_directory / "data.json").write_text(json.dumps(dict(
        cont_areas=result.cont_area_data,
        class_counts=result.class_counts,
        districts=result.district_data,
        territories=result.territory_data,
        provinces=result.province_data,
    )))

def main_districts_from_areas(regenerate_areas: bool = False) -> None:
    input_directory = Path(__file__).parent / "examples" / "input"
    output_directory = Path(__file__).parent / "examples" / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    cont_area_image_path = output_directory / "cont_area_image.png"
    cont_area_data_path = output_directory / "cont_area_data.json"

    maptool = MapTool(
        land_image=Image.open(input_directory / "land2.png"),
        boundary_image=Image.open(input_directory / "bound2_edited.png"),
    )

    if regenerate_areas:
        cont_area_image, cont_area_image_buffer, cont_area_data = maptool._generate_cont_areas()
        cont_area_image.save(cont_area_image_path)
        cont_area_data_path.write_text(json.dumps(cont_area_data))
    else:
        if not cont_area_image_path.exists() or not cont_area_data_path.exists():
            raise FileNotFoundError(
                "Missing precomputed area files. Expected: "
                f"{cont_area_image_path} and {cont_area_data_path}"
            )

        cont_area_image_buffer = np.array(Image.open(cont_area_image_path).convert("RGBA"), dtype=np.uint8)
        cont_area_data = json.loads(cont_area_data_path.read_text())

    class_image, class_image_buffer, class_counts = maptool._generate_type_classification()
    district_image, _, district_data = maptool._generate_districts(
        cont_area_image=cont_area_image_buffer,
        cont_area_data=cont_area_data,
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


def main_selective_steps(generate_steps=None, regenerate_areas=False):
    """
    Entrypoint to selectively generate steps (cont_areas, districts, territories, provinces).
    Steps not specified are loaded from files if they exist.
    """
    input_directory = Path(__file__).parent / "examples" / "input"
    output_directory = Path(__file__).parent / "examples" / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Step file paths
    paths = {
        "cont_areas": {
            "image": output_directory / "cont_area_image.png",
            "data": output_directory / "cont_area_data.json",
        },
        "class": {
            "image": output_directory / "class_image.png",
        },
        "districts": {
            "image": output_directory / "district_image.png",
            "data": output_directory / "district_data.json",
        },
        "territories": {
            "image": output_directory / "territory_image.png",
            "data": output_directory / "territory_data.json",
        },
        "provinces": {
            "image": output_directory / "province_image.png",
            "data": output_directory / "province_data.json",
        },
    }

    maptool = MapTool(
        land_image=Image.open(input_directory / "land2.png"),
        boundary_image=Image.open(input_directory / "bound2_edited.png"),
    )

    # Load or generate cont_areas
    if "cont_areas" in generate_steps or regenerate_areas:
        cont_area_image, cont_area_image_buffer, cont_area_data = maptool._generate_cont_areas()
        cont_area_image.save(paths["cont_areas"]["image"])
        paths["cont_areas"]["data"].write_text(json.dumps(cont_area_data))
    else:
        if not paths["cont_areas"]["image"].exists() or not paths["cont_areas"]["data"].exists():
            raise FileNotFoundError("Missing precomputed area files.")
        cont_area_image_buffer = np.array(Image.open(paths["cont_areas"]["image"]).convert("RGBA"), dtype=np.uint8)
        cont_area_data = json.loads(paths["cont_areas"]["data"].read_text())

    # Load or generate class image
    class_image, class_image_buffer, class_counts = maptool._generate_type_classification()
    class_image.save(paths["class"]["image"])

    # Load or generate districts
    if "districts" in generate_steps:
        district_image, _, district_data = maptool._generate_districts(
            cont_area_image=cont_area_image_buffer,
            cont_area_data=cont_area_data,
            class_image=class_image_buffer,
            class_counts=class_counts,
        )
        district_image.save(paths["districts"]["image"])
        paths["districts"]["data"].write_text(json.dumps(district_data))
    else:
        if paths["districts"]["image"].exists() and paths["districts"]["data"].exists():
            district_image = Image.open(paths["districts"]["image"])
            district_data = json.loads(paths["districts"]["data"].read_text())
        else:
            district_image, _, district_data = maptool._generate_districts(
                cont_area_image=cont_area_image_buffer,
                cont_area_data=cont_area_data,
                class_image=class_image_buffer,
                class_counts=class_counts,
            )
            district_image.save(paths["districts"]["image"])
            paths["districts"]["data"].write_text(json.dumps(district_data))

    # Repeat for territories
    if "territories" in generate_steps:
        territory_image, _, territory_data = maptool._generate_territories(
            district_image=np.array(district_image.convert("RGBA"), dtype=np.uint8),
            district_data=district_data,
        )
        territory_image.save(paths["territories"]["image"])
        paths["territories"]["data"].write_text(json.dumps(territory_data))
    else:
        if paths["territories"]["image"].exists() and paths["territories"]["data"].exists():
            territory_image = Image.open(paths["territories"]["image"])
            territory_data = json.loads(paths["territories"]["data"].read_text())
        else:
            territory_image, _, territory_data = maptool._generate_territories(
                district_image=np.array(district_image.convert("RGBA"), dtype=np.uint8),
                district_data=district_data,
                class_image=class_image_buffer,
                class_counts=class_counts,
            )
            territory_image.save(paths["territories"]["image"])
            paths["territories"]["data"].write_text(json.dumps(territory_data))

    # Repeat for provinces
    if "provinces" in generate_steps:
        province_image, _, province_data = maptool._generate_provinces(
            territory_image=np.array(territory_image.convert("RGBA"), dtype=np.uint8),
            territory_data=territory_data,
        )
        province_image.save(paths["provinces"]["image"])
        paths["provinces"]["data"].write_text(json.dumps(province_data))
    else:
        if paths["provinces"]["image"].exists() and paths["provinces"]["data"].exists():
            province_image = Image.open(paths["provinces"]["image"])
            province_data = json.loads(paths["provinces"]["data"].read_text())
        else:
            province_image, _, province_data = maptool._generate_provinces(
                territory_image=np.array(territory_image.convert("RGBA"), dtype=np.uint8),
                territory_data=territory_data,
                class_image=class_image_buffer,
                class_counts=class_counts,
            )
            province_image.save(paths["provinces"]["image"])
            paths["provinces"]["data"].write_text(json.dumps(province_data))

    # Save summary data
    (output_directory / "data.json").write_text(json.dumps(dict(
        cont_areas=cont_area_data,
        class_counts=class_counts,
        districts=district_data,
        territories=territory_data,
        provinces=province_data,
    )))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenGS MapTool entrypoints")
    parser.add_argument("-gui", action="store_true", help="Launch the GUI")
    parser.add_argument("-steps", nargs="*", default=[], help="Steps to generate: cont_areas, districts, territories, provinces")
    parser.add_argument("-regenerate-areas", action="store_true", help="Regenerate continuous areas before other steps")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.gui:
        main_gui()
    elif args.steps:
        main_selective_steps(generate_steps=args.steps, regenerate_areas=args.regenerate_areas)
    else:
        main_automatic()
