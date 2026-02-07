from pathlib import Path
from PIL import Image
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QTabWidget, QLabel, QCheckBox
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt, QUrl
from logic.maptool import MapTool
from ui.buttons import create_slider, create_button
from ui.image_display import ImageDisplay
import config


EXAMPLE_INPUT_DIR = Path(__file__).parent.parent / "example_input"
EXAMPLE_BOUNDARY_ORIG_IMAGE = Image.open(EXAMPLE_INPUT_DIR / "bound2_orig.png")
EXAMPLE_LAND_IMAGE = Image.open(EXAMPLE_INPUT_DIR / "land2.png")
EMPTY_IMAGE = Image.new("RGB", EXAMPLE_BOUNDARY_ORIG_IMAGE.size, color=(100, 100, 100))

class MapToolWindow(QWidget):
    """
    Open Grand Strategy Map Tool, which can be used from a UI Window.
    """

    def __init__(self) -> None:
        super().__init__()
        self.create_layout()
        self.showMaximized()
    
    def create_layout(self) -> None:
        # MAIN LAYOUT
        self.setWindowTitle(config.TITLE)
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        #self.progress = QProgressBar()
        #self.progress.setVisible(False)
        #main_layout.addWidget(self.progress)
        #self.progress.setMinimum(0)
        #self.progress.setMaximum(100)
        #self.progress.setValue(0)

        self.label_version = QLabel("Version "+config.VERSION)
        main_layout.addWidget(self.label_version)

        self.create_readme_tab()
        self.tabs.addTab(self.readme_tab, "README")
        self.create_boundary_tab()
        self.tabs.addTab(self.boundary_tab, "Adapt Boundary Image")
        self.create_input_images_tab()
        self.tabs.addTab(self.land_tab, "Input Images")
        self.create_province_tab()
        self.tabs.addTab(self.province_tab, "Province Image")
        self.create_territory_tab()
        self.tabs.addTab(self.territory_tab, "Territory Image")

    def create_readme_tab(self) -> None:
        self.readme_tab = QWidget()
        readme_layout = QVBoxLayout(self.readme_tab)
        self.readme_label = QLabel(
            '<h2>Please read the README</h2>' # TODO: change to Thomas-Holtvedt
            '<p><a href="https://github.com/GermanCodeEngineer/opengs-maptool/blob/main/README.md">'
            'Open the README in your browser</a></p>'
        )
        self.readme_label.setOpenExternalLinks(True)
        self.readme_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        readme_layout.addWidget(self.readme_label)

    def create_boundary_tab(self) -> None:
        self.boundary_tab = QWidget()
        boundary_tab_layout = QVBoxLayout(self.boundary_tab)
        
        create_button(boundary_tab_layout, "Import Boundary Image", self.on_button_import_boundary)

        self.orig_boundary_image_display = ImageDisplay(name="Boundary Image")
        self.orig_boundary_image_display.set_image(EXAMPLE_BOUNDARY_ORIG_IMAGE)
        boundary_tab_layout.addWidget(self.orig_boundary_image_display, stretch=1)

        create_button(boundary_tab_layout,
            "Normalize Territory and Province Density",
            self.on_button_normalize_density,
        )
        
        self.normalized_boundary_image_display = ImageDisplay(name="Adapted Boundary Image")
        self.normalized_boundary_image_display.set_image(EMPTY_IMAGE)
        boundary_tab_layout.addWidget(self.normalized_boundary_image_display, stretch=1)


    def create_input_images_tab(self) -> None:
        self.land_tab = QWidget()
        land_tab_layout = QVBoxLayout(self.land_tab)

        boundary_button_row = QHBoxLayout()
        land_tab_layout.addLayout(boundary_button_row)
        create_button(boundary_button_row, "Import Final Boundary Image", self.on_button_import_final_boundary)
        create_button(boundary_button_row, "Keep Generated Boundary Image", self.on_button_keep_generated_boundary)
        
        self.final_boundary_image_display = ImageDisplay(name="Final Boundary Image")
        self.final_boundary_image_display.set_image(EMPTY_IMAGE)
        land_tab_layout.addWidget(self.final_boundary_image_display, stretch=1)

        create_button(land_tab_layout, "Import Land Image", self.on_button_import_land)
        self.land_image_display = ImageDisplay(name="Land Image")
        self.land_image_display.set_image(EMPTY_IMAGE)
        land_tab_layout.addWidget(self.land_image_display, stretch=1)

    def create_province_tab(self) -> None:
        # TAB3 PROVINCE IMAGE
        self.province_tab = QWidget()
        self.province_image_display = ImageDisplay(name="Province Image")
        self.province_image_display.set_image(EMPTY_IMAGE)
        province_tab_layout = QVBoxLayout(self.province_tab)
        province_tab_layout.addWidget(self.province_image_display, stretch=1)
        button_row = QHBoxLayout()
        province_tab_layout.addLayout(button_row)

        # Buttons
        self.pixels_per_land_province_slider = create_slider(province_tab_layout,
            "Pixels per Land province:",
            config.PIXELS_PER_LAND_PROVINCE_MIN,
            config.PIXELS_PER_LAND_PROVINCE_MAX,
            config.PIXELS_PER_LAND_PROVINCE_DEFAULT,
            config.PIXELS_PER_LAND_PROVINCE_TICK,
            config.PIXELS_PER_LAND_PROVINCE_STEP,
        )

        self.pixels_per_water_province_slider = create_slider(province_tab_layout,
            "Pixels per Water province:",
            config.PIXELS_PER_WATER_PROVINCE_MIN,
            config.PIXELS_PER_WATER_PROVINCE_MAX,
            config.PIXELS_PER_WATER_PROVINCE_DEFAULT,
            config.PIXELS_PER_WATER_PROVINCE_TICK,
            config.PIXELS_PER_WATER_PROVINCE_STEP,
        )

        self.button_gen_prov = create_button(province_tab_layout,
            "Generate Provinces",
            self.on_button_generate_provinces,
        )
        #self.button_gen_prov.setEnabled(False)

        self.button_exp_prov_img = create_button(button_row,
            "Export Province Map",
            self.on_button_export_province_image,
        )
        #self.button_exp_prov_img.setEnabled(False)

        self.button_exp_prov_csv = create_button(button_row,
            "Export Province CSV",
            self.on_button_export_province_csv,
        )
        #self.button_exp_prov_csv.setEnabled(False)

    def create_territory_tab(self) -> None:
        # TAB4 TERRITORY IMAGE
        self.territory_tab = QWidget()
        self.territory_image_display = ImageDisplay(name="Territory Image")
        self.territory_image_display.set_image(EMPTY_IMAGE)
        territory_tab_layout = QVBoxLayout(self.territory_tab)
        territory_tab_layout.addWidget(self.territory_image_display, stretch=1)
        button_territory_row = QHBoxLayout()
        territory_tab_layout.addLayout(button_territory_row)

        # Buttons
        self.pixels_per_land_territory_slider = create_slider(territory_tab_layout,
            "Territory Land Density:",
            config.PIXELS_PER_LAND_TERRITORY_MIN,
            config.PIXELS_PER_LAND_TERRITORY_MAX,
            config.PIXELS_PER_LAND_TERRITORY_DEFAULT,
            config.PIXELS_PER_LAND_TERRITORY_TICK,
            config.PIXELS_PER_LAND_TERRITORY_STEP,
        )

        self.pixels_per_water_territory_slider = create_slider(territory_tab_layout,
            "Territory Ocean Density:",
            config.PIXELS_PER_WATER_TERRITORY_MIN,
            config.PIXELS_PER_WATER_TERRITORY_MAX,
            config.PIXELS_PER_WATER_TERRITORY_DEFAULT,
            config.PIXELS_PER_WATER_TERRITORY_TICK,
            config.PIXELS_PER_WATER_TERRITORY_STEP,
        )

        self.button_gen_territories = create_button(territory_tab_layout,
            "Generate Territories",
            self.on_button_generate_territories,
        )
        #self.button_gen_territories.setEnabled(False)

        self.button_exp_terr_img = create_button(button_territory_row,
            "Export Territory Map",
            self.on_button_export_territory_image,
        )
        #self.button_exp_terr_img.setEnabled(False)

        self.button_exp_terr_csv = create_button(button_territory_row,
            "Export Territory CSV",
            self.on_button_export_territory_csv,
        )
        #self.button_exp_terr_csv.setEnabled(False)

        self.button_exp_terr_json = create_button(button_territory_row,
            "Export Territory JSON",
            self.on_button_export_territory_json,
        )
        #self.button_exp_terr_json.setEnabled(False)


    # TAB 1
    def on_button_import_boundary(self) -> None:
        self.orig_boundary_image_display.import_image()

    def on_button_normalize_density(self) -> None:
        image_buffer = self.orig_boundary_image_display.get_image_buffer()
        if image_buffer is not None:
            normalized_buffer = MapTool.normalize_boundary_area_density(image_buffer)
            self.normalized_boundary_image_display.set_image(Image.fromarray(normalized_buffer))

    # TAB 2
    def on_button_import_final_boundary(self) -> None:
        self.final_boundary_image_display.import_image()

    def on_button_keep_generated_boundary(self) -> None:
        self.final_boundary_image_display.set_image(self.normalized_boundary_image_display.get_image() or EMPTY_IMAGE)

    def on_button_import_land(self) -> None:
        self.land_image_display.import_image()

    # TAB 3
    def on_button_generate_provinces(self) -> None:
        # TODO: Implement province generation
        pass

    def on_button_export_province_image(self) -> None:
        # TODO: Implement province image export
        pass

    def on_button_export_province_csv(self) -> None:
        # TODO: Implement province CSV export
        pass

    # TAB 4
    def on_button_generate_territories(self) -> None:
        # TODO: Implement territory generation
        pass

    def on_button_export_territory_image(self) -> None:
        # TODO: Implement territory image export
        pass

    def on_button_export_territory_csv(self) -> None:
        # TODO: Implement territory CSV export
        pass

    def on_button_export_territory_json(self) -> None:
        # TODO: Implement territory JSON export
        pass

    def on_readme_load_finished(self, ok: bool) -> None:
        if ok:
            self.readme_view.show()
            self.readme_label.hide()
        else:
            self.readme_view.hide()
            self.readme_label.show()


    def generate(self) -> None:
        # WORK IN PROGRESS
        maptool = MapTool(
            land_image=Image.open(EXAMPLE_INPUT_DIR / "bound2_density.png"),
            boundary_image=Image.open(EXAMPLE_INPUT_DIR / "land2.png"),
        )
