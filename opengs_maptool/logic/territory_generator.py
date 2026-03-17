from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from opengs_maptool.ui.main_window import MainWindow

import opengs_maptool.config as config
import numpy as np
from opengs_maptool.logic.numb_gen import NumberSeries
from opengs_maptool.logic.utils import (
    clear_used_colors, extract_masks, create_region_map, combine_maps,
    make_progress_updater, STEPS_PER_REGION_MAP, step
)


def generate_territory_map(main_layout: MainWindow) -> None:
    clear_used_colors()
    main_layout.progress.setVisible(True)
    main_layout.progress.setValue(0)

    boundary_image = main_layout.boundary_image_display.get_image()
    land_image = main_layout.land_image_display.get_image()

    masks = extract_masks(boundary_image, land_image)

    series = NumberSeries(
        config.TERRITORY_ID_PREFIX,
        config.TERRITORY_ID_START,
        config.TERRITORY_ID_END
    )

    density_arr = np.array(main_layout.density_image)
    density_strength = main_layout.territory_density_strength.value() / 10.0
    exclude_ocean_density = main_layout.territory_exclude_ocean_density.isChecked()
    jagged_land = main_layout.territory_jagged_land.isChecked()
    jagged_ocean = main_layout.territory_jagged_ocean.isChecked()

    land_points = main_layout.territory_land_slider.value()
    sea_points = main_layout.territory_ocean_slider.value()
    has_sea = sea_points > 0 and land_image is not None

    sea_step_budget = STEPS_PER_REGION_MAP if has_sea else 2
    total_steps = 2 + STEPS_PER_REGION_MAP + sea_step_budget + 2
    step = make_progress_updater(main_layout, total_steps)
    step(2)  # setup complete

    land_map, land_meta, next_index = create_region_map(
        masks["land_fill"], masks["land_border"], land_points, 0,
        "land", series, "territory_id", "territory_type", step_fn=step,
        density=density_arr, density_strength=density_strength,
        jagged=jagged_land
    )

    sea_density = None if exclude_ocean_density else density_arr
    sea_density_strength = 1.0 if exclude_ocean_density else density_strength

    if has_sea:
        sea_map, sea_meta, _ = create_region_map(
            masks["sea_fill"], masks["sea_border"], sea_points, next_index,
            "ocean", series, "territory_id", "territory_type", step_fn=step,
            density=sea_density, density_strength=sea_density_strength,
            jagged=jagged_ocean
        )
    else:
        sea_map = np.full((masks["map_h"], masks["map_w"]), -1, np.int32)
        sea_meta = []
        step(2)

    metadata = land_meta + sea_meta

    territory_image, combined_pmap = combine_maps(
        land_map, sea_map, metadata, masks["land_mask"], masks["sea_mask"]
    )
    step(1)

    main_layout.territory_image_display.set_image(territory_image)
    main_layout.territory_data = metadata
    main_layout.territory_pmap = combined_pmap
    main_layout.cached_masks = masks
    step(1)

    main_layout.progress.setValue(100)

    # Enable province generation and territory image export
    main_layout.button_gen_prov.setEnabled(True)
    main_layout.button_exp_terr_img.setEnabled(True)
    main_layout.button_exp_terr_def.setEnabled(True)

    # Reset province state if re-generating territories
    main_layout.province_data = None
    main_layout.button_exp_prov_img.setEnabled(False)
    main_layout.button_exp_prov_def.setEnabled(False)
    main_layout.button_exp_terr_hist.setEnabled(False)

    return territory_image, metadata
