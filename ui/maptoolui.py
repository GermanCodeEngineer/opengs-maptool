from pathlib import Path
from PIL import Image
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QTabWidget, QLabel
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt
from logic.maptool import MapTool
from ui.buttons import create_slider, create_button
from ui.image_display import ImageDisplay
import config


EXAMPLE_INPUT_DIR = Path(__file__).parent.parent / "example_input"
EXAMPLE_BOUNDARY_IMAGE = Image.open(EXAMPLE_INPUT_DIR / "bound2_density.png")
EXAMPLE_LAND_IMAGE = Image.open(EXAMPLE_INPUT_DIR / "land2.png")

class MapToolWindow(QWidget):
    """
    Open Grand Strategy Map Tool, which can be used from a UI Window.
    """
    maptool: MapTool

    def __init__(self) -> None:
        super().__init__()
        self.create_layout()
    
    def create_layout(self) -> None:
        # MAIN LAYOUT
        self.setWindowTitle(config.TITLE)
        self.resize(config.WINDOW_SIZE_WIDTH,
                    config.WINDOW_SIZE_HEIGHT)
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

        self.create_boundary_tab()
        self.create_land_tab()
        self.create_province_tab()
        self.create_territory_tab()

    def create_boundary_tab(self) -> None:
        # TAB1 BOUNDARY IMAGE
        self.boundary_tab = QWidget()
        boundary_tab_layout = QVBoxLayout(self.boundary_tab)
        
        # Documentation link at top
        self.label_docs = QLabel(
            '<a href="https://github.com/GermanCodeEngineer/opengs-maptool/blob/main/README.md">'
            '<h1>Please Read the Documentation First</h1></a>')
        self.label_docs.setOpenExternalLinks(True)
        docs_font = self.label_docs.font()
        docs_font.setUnderline(True)
        self.label_docs.setFont(docs_font)
        self.label_docs.setStyleSheet("color: #0066cc;")
        boundary_tab_layout.addWidget(self.label_docs)
        
        self.boundary_image_display = ImageDisplay()
        self.boundary_image_display.set_image(EXAMPLE_BOUNDARY_IMAGE)
        boundary_tab_layout.addWidget(self.boundary_image_display)
        self.tabs.addTab(self.boundary_tab, "Boundary Image")
        # HERE

        # Buttons
        create_button(boundary_tab_layout,
            "Import Boundary Image",
            lambda: None,
            #lambda: import_image(self,
            #    "Import Boundary Image",
            #    self.boundary_image_display
            #)
        )

    def create_land_tab(self) -> None:
        # TAB2 LAND IMAGE
        self.land_tab = QWidget()
        self.land_image_display = ImageDisplay()
        land_tab_layout = QVBoxLayout(self.land_tab)
        land_tab_layout.addWidget(self.land_image_display)
        self.tabs.addTab(self.land_tab, "Land Image")

        # Buttons
        create_button(land_tab_layout,
            "Import Land Image",
            lambda: None,
            #lambda: import_image(self,
            #                    "Import Land Image",
            #                    self.land_image_display
            #)
        )

    def create_province_tab(self) -> None:
        # TAB3 PROVINCE IMAGE
        self.province_tab = QWidget()
        self.province_image_display = ImageDisplay()
        province_tab_layout = QVBoxLayout(self.province_tab)
        province_tab_layout.addWidget(self.province_image_display)
        self.tabs.addTab(self.province_tab, "Province Image")
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
            lambda: None,
            #lambda: generate_province_map(self)
        )
        self.button_gen_prov.setEnabled(False)

        self.button_exp_prov_img = create_button(button_row,
            "Export Province Map",
            lambda: None,
            #lambda: export_image(self,
            #                    self.province_image_display.get_image(),
            #                    "Export Province Map"
            #)
        )
        self.button_exp_prov_img.setEnabled(False)

        self.button_exp_prov_csv = create_button(button_row,
            "Export Province CSV",
            lambda: None,
            #lambda: export_provinces_csv(self)
        )
        self.button_exp_prov_csv.setEnabled(False)

    def create_territory_tab(self) -> None:
        # TAB4 TERRITORY IMAGE
        self.territory_tab = QWidget()
        self.territory_image_display = ImageDisplay()
        territory_tab_layout = QVBoxLayout(self.territory_tab)
        territory_tab_layout.addWidget(self.territory_image_display)
        self.tabs.addTab(self.territory_tab, "Territory Image")
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
            lambda: None,
            #lambda: generate_territory_map(self),
        )
        self.button_gen_territories.setEnabled(False)

        self.button_exp_terr_img = create_button(button_territory_row,
            "Export Territory Map",
            lambda: None,
            #lambda: export_image(self,
            #    self.territory_image_display.get_image(),
            #    "Export Territory Map"
            #)
        )
        self.button_exp_terr_img.setEnabled(False)

        self.button_exp_terr_csv = create_button(button_territory_row,
            "Export Territory CSV",
            lambda: None,
            #lambda: export_territories_csv(self),
        )
        self.button_exp_terr_csv.setEnabled(False)

        self.button_exp_terr_json = create_button(button_territory_row,
            "Export Territory JSON",
            lambda: None,
            #lambda: export_territories_json(self),
        )
        self.button_exp_terr_json.setEnabled(False)


    def generate(self) -> None:
        # WORK IN PROGRESS
        maptool = MapTool(
            land_image=Image.open(EXAMPLE_INPUT_DIR / "bound2_density.png"),
            boundary_image=Image.open(EXAMPLE_INPUT_DIR / "land2.png"),
        )
