from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from opengs_maptool.ui.main_window import MainWindow

import opengs_maptool.config as config
import numpy as np
from PIL import Image


def normalize_density(layout: MainWindow) -> None:
    land_image = layout.land_image_display.get_image()
    if land_image is None:
        return

    w, h = land_image.size
    density = Image.new("L", (w, h), config.DEFAULT_DENSITY_GREY)
    layout.density_image = density

    layout.density_image_display.set_image(density.convert("RGBA"))
    layout.check_territory_ready()


def equator_density(layout: MainWindow) -> None:
    land_image = layout.land_image_display.get_image()
    if land_image is None:
        return

    w, h = land_image.size
    # Black (0) at equator (middle row), white (255) at top/bottom poles
    rows = np.linspace(0, 1, h)
    gradient = np.abs(rows - 0.5) * 2.0  # 0 at center, 1 at edges
    pixel_values = (gradient * 255).astype(np.uint8)
    arr = np.tile(pixel_values[:, np.newaxis], (1, w))

    density = Image.fromarray(arr, mode="L")
    layout.density_image = density
    layout.density_image_display.set_image(density.convert("RGBA"))
    layout.check_territory_ready()
