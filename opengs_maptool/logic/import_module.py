from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from opengs_maptool.ui.main_window import MainWindow
    from opengs_maptool.ui.image_display import ImageDisplay

import opengs_maptool.config as config
from PIL import Image
from PyQt6.QtWidgets import QFileDialog


def import_image(layout: MainWindow, text: str, image_display: ImageDisplay) -> None:
    Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
    path, _ = QFileDialog.getOpenFileName(
        layout,
        text,
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
    )
    if not path:
        return

    imported_image = Image.open(path).convert("RGBA")
    image_display.set_image(imported_image)

    # When importing a new land image, reset density (dimensions may differ)
    if image_display is layout.land_image_display:
        layout.density_image = None
        layout.density_image_display.set_image(None)
        layout.button_normalize_density.setEnabled(True)
        layout.button_equator_density.setEnabled(True)

    layout.check_territory_ready()


def import_terrain_image(layout: MainWindow) -> None:
    Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
    path, _ = QFileDialog.getOpenFileName(
        layout,
        "Import Terrain Image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
    )
    if not path:
        return

    terrain = Image.open(path).convert("RGB")
    layout.terrain_image = terrain
    layout.terrain_image_display.set_image(terrain.convert("RGBA"))


def import_density_image(layout: MainWindow) -> None:
    Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
    path, _ = QFileDialog.getOpenFileName(
        layout,
        "Import Density Image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
    )
    if not path:
        return

    density = Image.open(path).convert("L")
    layout.density_image = density

    layout.density_image_display.set_image(density.convert("RGBA"))
    layout.check_territory_ready()
