import opengs_maptool.config as config
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QTabWidget, QLabel
from opengs_maptool.logic.province_generator import generate_province_map
from opengs_maptool.logic.territory_generator import generate_territory_map
from opengs_maptool.logic.import_module import import_image, import_density_image, import_terrain_image
from opengs_maptool.logic.density_generator import normalize_density, equator_density
from opengs_maptool.logic.export_module import (export_image, export_territory_definitions,
                                 export_territory_history,
                                 export_province_definitions)
from opengs_maptool.ui.buttons import create_slider, create_button, create_checkbox
from opengs_maptool.ui.image_display import ImageDisplay


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # MAIN LAYOUT
        self.setWindowTitle(config.TITLE)
        self.setMinimumSize(800, 600)
        self.resize(config.WINDOW_SIZE_WIDTH,
                    config.WINDOW_SIZE_HEIGHT)
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)

        self.label_version = QLabel("Version "+config.VERSION)
        main_layout.addWidget(self.label_version)

        self.create_land_tab()
        self.create_boundary_tab()
        self.create_density_tab()
        self.create_terrain_tab()
        self.create_territory_tab()
        self.create_province_tab()

    def create_land_tab(self):
        self.land_tab = QWidget()
        self.land_image_display = ImageDisplay()
        land_tab_layout = QVBoxLayout(self.land_tab)
        land_tab_layout.addWidget(self.land_image_display)
        self.tabs.addTab(self.land_tab, "Land Image")
        create_button(land_tab_layout,
                      "Import Land Image",
                      lambda: import_image(self,
                                           "Import Land Image",
                                           self.land_image_display))
        self.density_image = None

    def create_boundary_tab(self):
        self.boundary_tab = QWidget()
        self.boundary_image_display = ImageDisplay()
        boundary_tab_layout = QVBoxLayout(self.boundary_tab)
        boundary_tab_layout.addWidget(self.boundary_image_display)
        self.tabs.addTab(self.boundary_tab, "Boundary Image")
        create_button(boundary_tab_layout,
                      "Import Boundary Image",
                      lambda: import_image(self,
                                           "Import Boundary Image",
                                           self.boundary_image_display))

    def create_density_tab(self):
        self.density_tab = QWidget()
        self.density_image_display = ImageDisplay()
        density_tab_layout = QVBoxLayout(self.density_tab)
        density_tab_layout.addWidget(self.density_image_display)
        self.tabs.addTab(self.density_tab, "Density Image")
        density_preset_row = QHBoxLayout()
        density_tab_layout.addLayout(density_preset_row)
        self.button_normalize_density = create_button(
            density_preset_row,
            "Normalize Density",
            lambda: normalize_density(self))
        self.button_normalize_density.setEnabled(False)
        self.button_equator_density = create_button(
            density_preset_row,
            "Equator Distribution",
            lambda: equator_density(self))
        self.button_equator_density.setEnabled(False)
        create_button(density_tab_layout,
                      "Import Density Image",
                      lambda: import_density_image(self))
        self.territory_exclude_ocean_density = create_checkbox(
            density_tab_layout, "Territory Exclude Ocean")
        self.province_exclude_ocean_density = create_checkbox(
            density_tab_layout, "Province Exclude Ocean")

    def create_terrain_tab(self):
        self.terrain_tab = QWidget()
        self.terrain_image_display = ImageDisplay()
        terrain_tab_layout = QVBoxLayout(self.terrain_tab)
        terrain_tab_layout.addWidget(self.terrain_image_display)
        self.tabs.addTab(self.terrain_tab, "Terrain Image")
        create_button(terrain_tab_layout,
                      "Import Terrain Image",
                      lambda: import_terrain_image(self))
        self.terrain_image = None

    def create_territory_tab(self):
        self.territory_tab = QWidget()
        self.territory_image_display = ImageDisplay()
        territory_tab_layout = QVBoxLayout(self.territory_tab)
        territory_tab_layout.addWidget(self.territory_image_display)
        self.tabs.addTab(self.territory_tab, "Territory Image")
        button_territory_row = QHBoxLayout()
        territory_tab_layout.addLayout(button_territory_row)
        self.territory_land_slider = create_slider(territory_tab_layout,
                                                   "Territory Land Density:",
                                                   config.LAND_TERRITORIES_MIN,
                                                   config.LAND_TERRITORIES_MAX,
                                                   config.LAND_TERRITORIES_DEFAULT,
                                                   config.LAND_TERRITORIES_TICK,
                                                   config.LAND_TERRITORIES_STEP)
        self.territory_ocean_slider = create_slider(territory_tab_layout,
                                                    "Territory Ocean Density:",
                                                    config.OCEAN_TERRITORIES_MIN,
                                                    config.OCEAN_TERRITORIES_MAX,
                                                    config.OCEAN_TERRITORIES_DEFAULT,
                                                    config.OCEAN_TERRITORIES_TICK,
                                                    config.OCEAN_TERRITORIES_STEP)

        territory_density_row = QHBoxLayout()
        territory_tab_layout.addLayout(territory_density_row)

        density_slider_col = QVBoxLayout()
        territory_density_row.addLayout(density_slider_col, stretch=1)

        self.territory_density_strength = create_slider(
            density_slider_col,
            "Density Strength:",
            config.DENSITY_STRENGTH_MIN,
            config.DENSITY_STRENGTH_MAX,
            config.DENSITY_STRENGTH_DEFAULT,
            config.DENSITY_STRENGTH_TICK,
            config.DENSITY_STRENGTH_STEP,
            display_scale=0.1)

        jagged_col = QVBoxLayout()
        territory_density_row.addLayout(jagged_col)

        self.territory_jagged_land = create_checkbox(
            jagged_col, "Jagged Land Borders")
        self.territory_jagged_ocean = create_checkbox(
            jagged_col, "Jagged Ocean Borders")

        self.button_gen_territories = create_button(territory_tab_layout,
                                                    "Generate Territories",
                                                    lambda: generate_territory_map(self))
        self.button_gen_territories.setEnabled(False)

        self.button_exp_terr_img = create_button(button_territory_row,
                                                 "Export Territory Image",
                                                 lambda: export_image(self,
                                                                      self.territory_image_display.get_image(),
                                                                      "Export Territory Image"))
        self.button_exp_terr_img.setEnabled(False)

        self.button_exp_terr_def = create_button(button_territory_row,
                                                 "Export Territory Definitions",
                                                 lambda: export_territory_definitions(self))
        self.button_exp_terr_def.setEnabled(False)

        self.button_exp_terr_hist = create_button(button_territory_row,
                                                  "Export Territory History",
                                                  lambda: export_territory_history(self))
        self.button_exp_terr_hist.setEnabled(False)

    def create_province_tab(self):
        # TAB5 PROVINCE IMAGE
        self.province_tab = QWidget()
        self.province_image_display = ImageDisplay()
        province_tab_layout = QVBoxLayout(self.province_tab)
        province_tab_layout.addWidget(self.province_image_display)
        self.tabs.addTab(self.province_tab, "Province Image")
        button_row = QHBoxLayout()
        province_tab_layout.addLayout(button_row)

        # Buttons
        self.land_slider = create_slider(province_tab_layout,
                                         "Land province Density:",
                                         config.LAND_PROVINCES_MIN,
                                         config.LAND_PROVINCES_MAX,
                                         config.LAND_PROVINCES_DEFAULT,
                                         config.LAND_PROVINCES_TICK,
                                         config.LAND_PROVINCES_STEP)

        self.ocean_slider = create_slider(province_tab_layout,
                                          "Ocean province Density",
                                          config.OCEAN_PROVINCES_MIN,
                                          config.OCEAN_PROVINCES_MAX,
                                          config.OCEAN_PROVINCES_DEFAULT,
                                          config.OCEAN_PROVINCES_TICK,
                                          config.OCEAN_PROVINCES_STEP)

        province_density_row = QHBoxLayout()
        province_tab_layout.addLayout(province_density_row)

        prov_density_slider_col = QVBoxLayout()
        province_density_row.addLayout(prov_density_slider_col, stretch=1)

        self.province_density_strength = create_slider(
            prov_density_slider_col,
            "Density Strength:",
            config.DENSITY_STRENGTH_MIN,
            config.DENSITY_STRENGTH_MAX,
            config.DENSITY_STRENGTH_DEFAULT,
            config.DENSITY_STRENGTH_TICK,
            config.DENSITY_STRENGTH_STEP,
            display_scale=0.1)

        prov_jagged_col = QVBoxLayout()
        province_density_row.addLayout(prov_jagged_col)

        self.province_jagged_land = create_checkbox(
            prov_jagged_col, "Jagged Land Borders")
        self.province_jagged_ocean = create_checkbox(
            prov_jagged_col, "Jagged Ocean Borders")

        self.button_gen_prov = create_button(province_tab_layout,
                                             "Generate Provinces",
                                             lambda: generate_province_map(self))
        self.button_gen_prov.setEnabled(False)

        self.button_exp_prov_img = create_button(button_row,
                                                 "Export Province Image",
                                                 lambda: export_image(self,
                                                                      self.province_image_display.get_image(),
                                                                      "Export Province Image"))
        self.button_exp_prov_img.setEnabled(False)

        self.button_exp_prov_def = create_button(button_row,
                                                  "Export Province Definitions",
                                                  lambda: export_province_definitions(self))
        self.button_exp_prov_def.setEnabled(False)

    def check_territory_ready(self):
        land_exists = self.land_image_display.get_image() is not None
        density_exists = self.density_image is not None
        self.button_gen_territories.setEnabled(land_exists and density_exists)

