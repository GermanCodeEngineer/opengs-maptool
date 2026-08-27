from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from opengs_maptool.context import ApplicationContext

from opengs_maptool.controllers.progress_controller import ProgressController
from opengs_maptool.controllers.task_controller import ThreadTaskSlot, ThreadTask, ThreadSlotOccupiedError
from opengs_maptool.logic.density_generator import normalize_density, equator_density, remove_density_image
from opengs_maptool.services.command_core import register_command, create_sync_task_waiter, FinalTaskStatus, CommandArgSpec
from opengs_maptool.models.command_response import CommandResponse
from opengs_maptool.models.message import MessageType

@register_command("density.image.remove", args=[
    CommandArgSpec(
        name="wait_for_completion", arg_type=bool, default=False,
        description="Whether to wait for the task to complete before allowing to run another command.",
    ),
])
def cmd_density_image_remove(context: ApplicationContext, wait_for_completion: bool) -> CommandResponse:
    """Removes the current density image."""
    from opengs_maptool.context import LimitedTaskContext
    print("started: cmd_density_image_remove")

    def temp_do(*args, **kwargs):
        print("started: temp_do")
        import time # GCE-TODO: remove temp
        print("first sleep")
        time.sleep(1)
        print("after first sleep")
        remove_density_image(*args, **kwargs)
        print("second sleep")
        time.sleep(1)
        print("finished: temp_do")

    if wait_for_completion:
        connect_signals, execute_wait = create_sync_task_waiter()
    else:
        def connect_signals(task: ThreadTask):
            pass

    try:
        context.task_controller.start_task(
            function=temp_do,
            title="Removing density image",
            slot=ThreadTaskSlot.change_density_image,
            provide_progress_controller=True,
            pos_args=[],
            kw_args={"task_ctx": LimitedTaskContext(context)},
            before_start_callback=connect_signals,
        )
    except ThreadSlotOccupiedError: # GCE-TODO: instead use if occupied slot check
        return CommandResponse("A task to remove the density image is already running.", MessageType.ERROR)
    if not wait_for_completion:
        return CommandResponse("Started removing density image.", MessageType.NORMAL)
    else:
        status, result, error = execute_wait()
        match status:
            case FinalTaskStatus.ERROR:
                return CommandResponse(f"Failed to remove density image: {str(error)}", MessageType.ERROR)
            case FinalTaskStatus.SUCCESS:
                return CommandResponse("Removed density image.", MessageType.NORMAL)
            case FinalTaskStatus.CANCELLED:
                return CommandResponse("Task was cancelled.", MessageType.ERROR)

@register_command("density.image.normalize", args=[
    CommandArgSpec(
        name="wait_for_completion", arg_type=bool, default=False,
        description="Whether to wait for the task to complete before allowing to run another command.",
    ),
])
def cmd_density_image_normalize(context: ApplicationContext, wait_for_completion: bool) -> CommandResponse:
    """Normalizes density values."""
    from opengs_maptool.context import LimitedTaskContext
    try:
        normalize_density(LimitedTaskContext(context), ProgressController())
        return CommandResponse(f"Normalized density values.", MessageType.NORMAL)
    except Exception as error:
        # There is no known possible error.
        return CommandResponse(f"Failed to normalize density values: {str(error)}", MessageType.ERROR)

@register_command("density.image.equator_distribute", args=[
    CommandArgSpec(
        name="wait_for_completion", arg_type=bool, default=False,
        description="Whether to wait for the task to complete before allowing to run another command.",
    ),
])
def cmd_density_image_equator_distribute(context: ApplicationContext, wait_for_completion: bool) -> CommandResponse:
    """Generates equator-based density distribution."""
    from opengs_maptool.context import LimitedTaskContext
    try:
        equator_density(LimitedTaskContext(context), ProgressController())
        return CommandResponse(f"Generated equator-based density values.", MessageType.NORMAL)
    except Exception as error:
        # There is no known possible error.
        return CommandResponse(f"Failed to generate equator-based density values: {str(error)}", MessageType.ERROR)
