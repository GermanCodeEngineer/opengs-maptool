import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt6.QtWidgets import QApplication
import sys

from . import MapToolWindow, MapTool, export_to_json, export_to_csv, import_from_json


def main_automatic() -> None:
    # Default paths
    input_directory = Path(__file__).parent / "examples" / "input"
    output_directory = Path(__file__).parent / "examples" / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    class StepMapTool(MapTool):
        def on_cont_areas_generated(self, cont_area_image, cont_area_image_buffer, cont_area_data):
            cont_area_image.save(output_directory / "cont_area_image.png")
            export_to_json(cont_area_data, output_directory / "cont_area_data.json")
        def on_dens_samps_generated(self, dens_samp_image, dens_samp_image_buffer, dens_samp_data):
            dens_samp_image.save(output_directory / "dens_samp_image.png")
            export_to_json(dens_samp_data, output_directory / "dens_samp_data.json")
        
        def on_territories_generated(self, territory_image, territory_image_buffer, territory_data):
            territory_image.save(output_directory / "territory_image.png")
            export_to_json(territory_data, output_directory / "territory_data.json")
        
        def on_provinces_generated(self, province_image, province_image_buffer, province_data):
            province_image.save(output_directory / "province_image.png")
            export_to_json(province_data, output_directory / "province_data.json")
            
    maptool = StepMapTool(
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
    window = MapToolWindow()
    window.show()
    sys.exit(app.exec())


def main_selective_steps(generate_steps=None, regenerate_areas=False, export_csv=False):
    """
    Entrypoint to selectively generate steps (cont_areas, dens_samps, territories, provinces).
    Steps not specified are loaded from files if they exist.
    """

    def export_formats(data, path: Path) -> None:
        export_to_json(data, path)
        if export_csv:
            export_to_csv(data, path)

    input_directory = Path(__file__).parent / "examples" / "input"
    output_directory = Path(__file__).parent / "examples" / "output"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Step file paths
    paths = {
        "cont_areas": {
            "image": output_directory / "cont_area_image.png",
            "data": output_directory / "cont_area_data.json",
        },
        "dens_samps": {
            "image": output_directory / "dens_samp_image.png",
            "data": output_directory / "dens_samp_data.json",
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
        class_image=MapTool.clean_class_image(Image.open(input_directory / "class2_clean.png")),
        boundary_image=Image.open(input_directory / "bound2_edited.png"),
    )

    # Load or generate cont_areas
    if "cont_areas" in generate_steps or regenerate_areas:
        cont_area_image, cont_area_image_buffer, cont_area_data = maptool._generate_cont_areas()
        cont_area_image.save(paths["cont_areas"]["image"])
        export_formats(cont_area_data, paths["cont_areas"]["data"])
        
    else:
        if not paths["cont_areas"]["image"].exists() or not paths["cont_areas"]["data"].exists():
            raise FileNotFoundError("Missing precomputed area files.")
        cont_area_image_buffer = np.array(Image.open(paths["cont_areas"]["image"]).convert("RGBA"), dtype=np.uint8)
        cont_area_data = import_from_json(paths["cont_areas"]["data"])

    # Load or generate dens_samps
    if "dens_samps" in generate_steps:
        dens_samp_image, _, dens_samp_data = maptool._generate_dens_samps(cont_area_image_buffer, cont_area_data)
        dens_samp_image.save(paths["dens_samps"]["image"])
        export_formats(dens_samp_data, paths["dens_samps"]["data"])
    else:
        if paths["dens_samps"]["image"].exists() and paths["dens_samps"]["data"].exists():
            dens_samp_image = Image.open(paths["dens_samps"]["image"])
            dens_samp_data = import_from_json(paths["dens_samps"]["data"])
        else:
            dens_samp_image, _, dens_samp_data = maptool._generate_dens_samps(cont_area_image_buffer, cont_area_data)
            dens_samp_image.save(paths["dens_samps"]["image"])
            export_formats(dens_samp_data, paths["dens_samps"]["data"])

    # Repeat for territories
    if "territories" in generate_steps:
        territory_image, _, territory_data = maptool._generate_territories(
            dens_samp_image=np.array(dens_samp_image.convert("RGBA"), dtype=np.uint8),
            dens_samp_data=dens_samp_data,
        )
        territory_image.save(paths["territories"]["image"])
        export_formats(territory_data, paths["territories"]["data"])
    else:
        if paths["territories"]["image"].exists() and paths["territories"]["data"].exists():
            territory_image = Image.open(paths["territories"]["image"])
            territory_data = import_from_json(paths["territories"]["data"])
        else:
            territory_image, _, territory_data = maptool._generate_territories(
                dens_samp_image=np.array(dens_samp_image.convert("RGBA"), dtype=np.uint8),
                dens_samp_data=dens_samp_data,
            )
            territory_image.save(paths["territories"]["image"])
            export_formats(territory_data, paths["territories"]["data"])

    # Repeat for provinces
    if "provinces" in generate_steps:
        province_image, _, province_data = maptool._generate_provinces(
            territory_image=np.array(territory_image.convert("RGBA"), dtype=np.uint8),
            territory_data=territory_data,
        )
        province_image.save(paths["provinces"]["image"])
        export_formats(province_data, paths["provinces"]["data"])
    else:
        if paths["provinces"]["image"].exists() and paths["provinces"]["data"].exists():
            province_image = Image.open(paths["provinces"]["image"])
            province_data = import_from_json(paths["provinces"]["data"])
        else:
            province_image, _, province_data = maptool._generate_provinces(
                territory_image=np.array(territory_image.convert("RGBA"), dtype=np.uint8),
                territory_data=territory_data,
            )
            province_image.save(paths["provinces"]["image"])
            export_formats(province_data, paths["provinces"]["data"])

    # Save summary data
    export_to_json(dict(
        cont_areas=cont_area_data,
        dens_samps=dens_samp_data,
        territories=territory_data,
        provinces=province_data,
    ), output_directory / "data.json")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenGS MapTool entrypoints")
    parser.add_argument("-gui", action="store_true", help="Launch the GUI")
    parser.add_argument("-steps", nargs="*", default=[], help="Steps to generate: cont_areas, dens_samps, territories, provinces")
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
