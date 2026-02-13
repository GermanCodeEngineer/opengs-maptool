import sys
from pathlib import Path
from PIL import Image
from PyQt6.QtWidgets import QApplication
from . import MapToolWindow, MapTool

def main_automatic() -> None:
    import json

    # Default paths
    input_directory = Path(__file__).parent / "examples" / "input"
    output_directory = Path(__file__).parent / "examples" / "output"

    class DistrictMapTool(MapTool):
        def on_cont_areas_generated(self, cont_areas_image, cont_areas_image_buffer, cont_areas_data):
            cont_areas_image.save(output_directory / "contareas_image.png")

        def on_districts_generated(self, district_image, district_image_buffer, district_data):
            district_image.save(output_directory / "district_image.png")
            sys.exit(0) # temporarily exit after that

    maptool = DistrictMapTool(
        land_image=Image.open(input_directory / "land2.png"),
        boundary_image=Image.open(input_directory / "bound2_orig_density.png"),#"bound2_density.png"),
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

def main_gui() -> None:
    app = QApplication(sys.argv)
    window = MapToolWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    if sys.argv[-1] == "-gui":
        main_gui()
    else:
        main_automatic()
