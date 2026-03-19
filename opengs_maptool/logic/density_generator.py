from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from opengs_maptool.ui.main_window import MainWindow

import opengs_maptool.config as config
import numpy as np
from PIL import Image


def normalize_density(main_layout: MainWindow) -> None:
    land_image = main_layout.get_land_image()
    if land_image is None:
        return

    w, h = land_image.size
    density = Image.new("L", (w, h), config.DEFAULT_DENSITY_GREY)
    main_layout.set_density_image(density)
    main_layout.check_territory_ready()


def equator_density(main_layout: MainWindow) -> None:
    land_image = main_layout.get_land_image()
    if land_image is None:
        return

    w, h = land_image.size
    # Black (0) at equator (middle row), white (255) at top/bottom poles
    rows = np.linspace(0, 1, h)
    gradient = np.abs(rows - 0.5) * 2.0  # 0 at center, 1 at edges
    pixel_values = (gradient * 255).astype(np.uint8)
    arr = np.tile(pixel_values[:, np.newaxis], (1, w))

    density = Image.fromarray(arr, mode="L")
    main_layout.set_density_image(density)
    main_layout.check_territory_ready()
