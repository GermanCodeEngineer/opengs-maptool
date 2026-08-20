from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from opengs_maptool.context import LimitedTaskContext

import opengs_maptool.config as config
import numpy as np
from PIL import Image
from opengs_maptool.controllers.progress_controller import ProgressController
from opengs_maptool.simple_types import TabName

class DensityGenerationNotAvailableError(RuntimeError):
    pass

def normalize_density(task_ctx: LimitedTaskContext, progress_controller: ProgressController) -> None:
    with progress_controller.execute_as_only_phase():
        project = task_ctx.project
        if not project.can_density_image_be_generated():
            return False

        w, h = project.land_image.size
        density_image = Image.new("L", (w, h), config.DEFAULT_DENSITY_GREY)

        project.density_image = density_image
        project.modified = True
        task_ctx.refresh_tab_view(TabName.DENSITY)
        return True

def equator_density(task_ctx: LimitedTaskContext, progress_controller: ProgressController) -> bool:
    phase1 = progress_controller.add_phase(step_weight=1, msg="Calculating density values")
    phase2 = progress_controller.add_phase(step_weight=1, msg="Assigning pixel values")
    phase3 = progress_controller.add_phase(step_weight=1, msg="Creating density image")

    with progress_controller.execute_phase(phase1):
        project = task_ctx.project
        if not project.can_density_image_be_generated():
            return False

        w, h = project.land_image.size

        # Black (0) at equator (middle row), white (255) at top/bottom poles
        rows = np.linspace(0, 1, h)
        gradient = np.abs(rows - 0.5) * 2.0  # 0 at center, 1 at edges

    with progress_controller.execute_phase(phase2):
        pixel_values = (gradient * 255).astype(np.uint8)
        arr = np.tile(pixel_values[:, np.newaxis], (1, w))

    with progress_controller.execute_phase(phase3):
        density_image = Image.fromarray(arr, mode="L")
        project.density_image = density_image
        project.modified = True
        task_ctx.refresh_tab_view(TabName.DENSITY)
        return True

# TODO: Move this function, it doesn't belong in this file
# -> if done, remove progress_controller argument
def remove_density_image(task_ctx: LimitedTaskContext, progress_controller: ProgressController) -> bool:
    # This is instant, no need for a progress bar
    project = task_ctx.project
    if not project.can_density_image_be_removed():
        return False

    project.density_image = None
    project.modified = True
    task_ctx.refresh_tab_view(TabName.DENSITY)
    return True
