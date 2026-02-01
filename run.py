import sys, pathlib, shutil
import config
from PIL import Image
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from logic.export_module import _export_provinces_csv_to, _export_territories_csv_to, _export_territories_json_to

input_directory = pathlib.Path(__file__).parent.parent.parent / "GodotProjects/opengs/wip/outputqgis"
output_directory = pathlib.Path(__file__).parent / "output"



if __name__ == "__main__":
    ilerp = lambda a, b, t: round(a + t * (b - a))
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    window.land_image_display.set_image(Image.open(input_directory / "landimage.png"))
    window.boundary_image_display.set_image(Image.open(input_directory / "boundaryimage.png"))
    window.button_gen_prov.setEnabled(True)
    
    window.land_slider.setValue(ilerp(config.LAND_PROVINCES_MIN, config.LAND_PROVINCES_MAX, 0.3))
    window.ocean_slider.setValue(ilerp(config.OCEAN_PROVINCES_MIN, config.OCEAN_PROVINCES_MAX, 0.3))
    window.territory_land_slider.setValue(ilerp(config.LAND_TERRITORIES_MIN, config.LAND_TERRITORIES_MAX, 0.7))
    window.territory_ocean_slider.setValue(ilerp(config.OCEAN_TERRITORIES_MIN, config.OCEAN_TERRITORIES_MAX, 0.4))

    output_directory.mkdir(parents=True, exist_ok=True)
    window.button_gen_prov.click()
    window.province_image_display.get_image().save(output_directory / "provinceimage.png")
    _export_provinces_csv_to(window.province_data, str(output_directory / "provincedata.csv"))

    window.button_gen_territories.click()
    window.territory_image_display.get_image().save(output_directory / "territoryimage.png")
    _export_territories_csv_to(window.territory_data, str(output_directory / "territorydata.csv"))
    
    _export_territories_json_to(window.territory_data, str(output_directory / "territorydata.json"))
    
    window.close()
    app.quit()
