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

    class_image = StepMapTool.clean_class_image(Image.open(input_directory / "class2_clean.png"))
    boundary_image = Image.open(input_directory / "bound2_edited.png")

    # Use StepMapTool for stepwise generation
    # Load or generate cont_areas
    if "cont_areas" in generate_steps or regenerate_areas:
        cont_area_image_buffer, cont_area_data = StepMapTool.generate_cont_areas(
            class_image, np.array(boundary_image.convert("RGBA")),
        )
        Image.fromarray(cont_area_image_buffer).save(paths["cont_areas"]["image"])
        export_formats(cont_area_data, paths["cont_areas"]["data"])
    else:
        if not paths["cont_areas"]["image"].exists() or not paths["cont_areas"]["data"].exists():
            raise FileNotFoundError("Missing precomputed area files.")
        cont_area_image_buffer = np.array(Image.open(paths["cont_areas"]["image"]).convert("RGBA"), dtype=np.uint8)
        cont_area_data = import_from_json(paths["cont_areas"]["data"])

    # Load or generate dens_samps
    if "dens_samps" in generate_steps:
        dens_samp_image_buffer, dens_samp_data = StepMapTool.generate_dens_samps(
            np.array(boundary_image.convert("RGBA")),
            cont_area_image_buffer,
            cont_area_data,
            pixels_per_land_dens_samp=1000,
            pixels_per_water_dens_samp=1000,
        )
        Image.fromarray(dens_samp_image_buffer).save(paths["dens_samps"]["image"])
        export_formats(dens_samp_data, paths["dens_samps"]["data"])
    else:
        if paths["dens_samps"]["image"].exists() and paths["dens_samps"]["data"].exists():
            dens_samp_image_buffer = np.array(Image.open(paths["dens_samps"]["image"]).convert("RGBA"), dtype=np.uint8)
            dens_samp_data = import_from_json(paths["dens_samps"]["data"])
        else:
            dens_samp_image_buffer, dens_samp_data = StepMapTool.generate_dens_samps(
                np.array(boundary_image.convert("RGBA")),
                cont_area_image_buffer,
                cont_area_data,
                pixels_per_land_dens_samp=1000,
                pixels_per_water_dens_samp=1000,
            )
            Image.fromarray(dens_samp_image_buffer).save(paths["dens_samps"]["image"])
            export_formats(dens_samp_data, paths["dens_samps"]["data"])

    # Repeat for territories
    if "territories" in generate_steps:
        territory_image_buffer, territory_data = StepMapTool.generate_territories(
            np.array(boundary_image.convert("RGBA")),
            dens_samp_image_buffer,
            dens_samp_data,
            pixels_per_land_territory=1000,
            pixels_per_water_territory=1000,
        )
        Image.fromarray(territory_image_buffer).save(paths["territories"]["image"])
        export_formats(territory_data, paths["territories"]["data"])
    else:
        if paths["territories"]["image"].exists() and paths["territories"]["data"].exists():
            territory_image_buffer = np.array(Image.open(paths["territories"]["image"]).convert("RGBA"), dtype=np.uint8)
            territory_data = import_from_json(paths["territories"]["data"])
        else:
            territory_image_buffer, territory_data = StepMapTool.generate_territories(
                np.array(boundary_image.convert("RGBA")),
                dens_samp_image_buffer,
                dens_samp_data,
                pixels_per_land_territory=1000,
                pixels_per_water_territory=1000,
            )
            Image.fromarray(territory_image_buffer).save(paths["territories"]["image"])
            export_formats(territory_data, paths["territories"]["data"])

    # Repeat for provinces
    if "provinces" in generate_steps:
        province_image_buffer, province_data = StepMapTool.generate_provinces(
            np.array(boundary_image.convert("RGBA")),
            territory_image_buffer,
            territory_data,
            pixels_per_land_province=1000,
            pixels_per_water_province=1000,
        )
        Image.fromarray(province_image_buffer).save(paths["provinces"]["image"])
        export_formats(province_data, paths["provinces"]["data"])
    else:
        if paths["provinces"]["image"].exists() and paths["provinces"]["data"].exists():
            province_image_buffer = np.array(Image.open(paths["provinces"]["image"]).convert("RGBA"), dtype=np.uint8)
            province_data = import_from_json(paths["provinces"]["data"])
        else:
            province_image_buffer, province_data = StepMapTool.generate_provinces(
                np.array(boundary_image.convert("RGBA")),
                territory_image_buffer,
                territory_data,
                pixels_per_land_province=1000,
                pixels_per_water_province=1000,
            )
            Image.fromarray(province_image_buffer).save(paths["provinces"]["image"])
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
