from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opengs_maptool.context import ApplicationContext

from opengs_maptool.controllers.progress_controller import ProgressController
from opengs_maptool.logic.density_generator import normalize_density, equator_density, remove_density_image
from opengs_maptool.services.command_core import register_command
from opengs_maptool.models.command_response import CommandResponse
from opengs_maptool.models.message import MessageType

@register_command("density.image.remove", args=[])
def cmd_density_image_remove(context: ApplicationContext) -> CommandResponse:
    """Removes the current density image."""
    from opengs_maptool.context import LimitedTaskContext
    try:
        remove_density_image(LimitedTaskContext(context), ProgressController())
        return CommandResponse(f"Removed density image.", MessageType.NORMAL)
    except Exception as error:
        # There is no known possible error.
        return CommandResponse(f"Failed to remove density image: {str(error)}", MessageType.ERROR)

@register_command("density.image.normalize", args=[])
def cmd_density_image_remove(context: ApplicationContext) -> CommandResponse:
    """Normalizes density values."""
    from opengs_maptool.context import LimitedTaskContext
    try:
        normalize_density(LimitedTaskContext(context), ProgressController())
        return CommandResponse(f"Normalized density values.", MessageType.NORMAL)
    except Exception as error:
        # There is no known possible error.
        return CommandResponse(f"Failed to normalize density values: {str(error)}", MessageType.ERROR)

@register_command("density.image.equator_distribute", args=[])
def cmd_density_image_remove(context: ApplicationContext) -> CommandResponse:
    """Generates equator-based density distribution."""
    from opengs_maptool.context import LimitedTaskContext
    try:
        equator_density(LimitedTaskContext(context), ProgressController())
        return CommandResponse(f"Generated equator-based density values.", MessageType.NORMAL)
    except Exception as error:
        # There is no known possible error.
        return CommandResponse(f"Failed to generate equator-based density values: {str(error)}", MessageType.ERROR)
