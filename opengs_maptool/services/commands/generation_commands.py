from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from opengs_maptool.context import ApplicationContext

from opengs_maptool.controllers.task_controller import ThreadTaskSlot, ThreadSlotOccupiedError
from opengs_maptool.services.command_core import register_command, CommandArgSpec
from opengs_maptool.logic.export_module import export_image
from opengs_maptool.logic.territory_generator import generate_territory_map
from opengs_maptool.logic.province_generator import generate_province_map
from opengs_maptool.models.command_response import CommandResponse
from opengs_maptool.models.message import MessageType

@register_command("territory.image.generate", args=[])
def cmd_territory_image_generate(context: ApplicationContext) -> CommandResponse:
    """Generates the territory map."""
    able, message = context.project.get_territory_image_generate_status(context, ignore_locked=False)
    if not able:
        return CommandResponse(message, MessageType.ERROR)
    return run_generation_task(
        context=context,
        task_function=generate_territory_map,
        task_slot=ThreadTaskSlot.generate_territory_map,
        task_title="Generating territory map",
        lowercase_title="generating territory map",
    )

@register_command("province.image.generate", args=[])
def cmd_province_image_generate(context: ApplicationContext) -> CommandResponse:
    """Generates the province map."""
    able, message = context.project.get_province_image_generate_status(context, ignore_locked=False)
    if not able:
        return CommandResponse(message, MessageType.ERROR)
    return run_generation_task(
        context=context,
        task_function=generate_province_map,
        task_slot=ThreadTaskSlot.generate_province_map,
        task_title="Generating province map",
        lowercase_title="generating province map",
    )

@register_command("territory.image.export", args=[
    CommandArgSpec("path", arg_type=str, description="The path to export the territory image to."),
])
def cmd_territory_image_export(context: ApplicationContext, path: str) -> CommandResponse:
    """Exports the territory image."""
    try:
        if not context.project.territory_image:
            return CommandResponse("No territory image to export.", MessageType.ERROR)
        export_image(path, context.project.territory_image, raise_on_error=True)
        return CommandResponse(f"Exported territory image to {path}.", MessageType.NORMAL)
    except Exception as error:
        return CommandResponse(f"Failed exporting territory image: {str(error)}", MessageType.ERROR)

@register_command("province.image.export", args=[
    CommandArgSpec("path", arg_type=str, description="The path to export the province image to."),
])
def cmd_province_image_export(context: ApplicationContext, path: str) -> CommandResponse:
    """Exports the province image."""
    if not context.project.province_image:
        return CommandResponse("No province image to export.", MessageType.ERROR)
    try:
        export_image(path, context.project.province_image, raise_on_error=True)
        return CommandResponse(f"Exported province image to {path}.", MessageType.NORMAL)
    except Exception as error:
        if isinstance(error, ValueError) and "unknown file extension" in str(error):
            return CommandResponse(f"Failed exporting province image: Invalid file extension", MessageType.ERROR)
        return CommandResponse(f"Failed exporting province image: {str(error)}", MessageType.ERROR)

def run_generation_task(context: ApplicationContext, task_function: Callable, task_slot: ThreadTaskSlot, task_title: str, lowercase_title: str):
    try:
        # Run in the proper ThreadTaskSlot that was created for the function
        # -> proper UI notification & task slot management
        context.task_controller.start_task(
            function=task_function,
            title=task_title,
            slot=task_slot,
            provide_progress_controller=True,
            pos_args=[],
            kw_args={
                "task_ctx": context,
            },
            before_start_callback=None,
        )
        return CommandResponse(f"Started {lowercase_title}...", MessageType.NORMAL)
    except ThreadSlotOccupiedError:
        return CommandResponse(f"A task {lowercase_title} is already running.", MessageType.ERROR)
    except Exception as error:
        # There is no known possible error.
        return CommandResponse(f"Failed {lowercase_title}: {str(error)}", MessageType.ERROR)
