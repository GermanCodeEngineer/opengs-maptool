from __future__ import annotations
from difflib import get_close_matches
import mslex
import asyncio
from enum import Enum
import re
import traceback
from typing import Callable, TypeAlias, Any, TYPE_CHECKING, Coroutine

if TYPE_CHECKING:
    from opengs_maptool.context import ApplicationContext

from opengs_maptool.controllers.task_controller import ThreadTaskSlot, ThreadTask
from opengs_maptool.models.command_response import CommandResponse
from opengs_maptool.models.message import MessageType
from opengs_maptool.services.parser_service import (
    CommandArgSpec,
    CommandArgumentParseError,
    CommandParserConfigurationError,
    deserialize_command_arguments,
)

#####################################################
#                Base Infrastructure                #
#####################################################

# Principle: The @decorator registers a command and the function docstring is used as the command description.

# command.this.format.id -> implementation function
CommandImplementation: TypeAlias = Callable[..., CommandResponse | Coroutine[Any, Any, CommandResponse]]
_commands: dict[str, tuple[CommandImplementation, str, list[CommandArgSpec]]] = {}
_command_aliases = (dict[str, str])() # alias -> real command
_SORT_PRIORITY_PREFIXES = ["link", "console", "project", "land", "boundary", "density", "terrain", "territory", "province"]
# /\ above: imported by other module

def register_command(
    command_id: str,
    args: list[CommandArgSpec],
    aliases: list[str] | None = None,
):
    """Decorator to register a command function with a given command ID."""
    def decorator[T: CommandImplementation](func: T) -> T:
        if command_exists(command_id):
            raise ValueError(f"Please report this. Command {command_id} is already registered.")

        doc = func.__doc__ or "No description provided"
        _commands[command_id] = (func, doc, args)

        if aliases:
            for alias in aliases:
                if not command_exists(command_id): # Practically unreachable as command is created above.
                    raise ValueError(f"Please report this. Cannot create alias '{alias}' for unknown command '{command_id}'")
                _command_aliases[alias] = command_id
        return func
    return decorator

def command_exists(command_id: str) -> bool:
    return command_id in _commands or command_id in _command_aliases

def get_command_implementation(command_id: str) -> CommandImplementation:
    command_id = _command_aliases.get(command_id, command_id)
    return _commands[command_id][0]

def get_command_description(command_id: str) -> str:
    command_id = _command_aliases.get(command_id, command_id)
    return _commands[command_id][1]

def get_command_arg_specs(command_id: str) -> list[CommandArgSpec]:
    command_id = _command_aliases.get(command_id, command_id)
    return _commands[command_id][2]

def get_all_command_ids() -> list[str]:
    return list(_commands.keys())

def get_all_command_aliases() -> list[str]:
    return list(_command_aliases.keys())

class FinalTaskStatus(Enum):
    ERROR = 1
    SUCCESS = 2
    CANCELLED = 3

# GCE-TODO: later convert this function into a class with private __init__ & public 2 methods
def create_async_task_waiter() -> tuple[Callable[[ThreadTask], None], Callable[[], Coroutine[Any, Any, tuple[FinalTaskStatus, Any, Any]]]]:
    """
    Creates a signal connector and an awaitable wait function.
    Attaches to signals inside `before_start_callback` to prevent race conditions.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[tuple[FinalTaskStatus, Any, Any]] = loop.create_future()

    def set_future_result(status: FinalTaskStatus, res: Any = None, err: Any = None):
        if not future.done():
            loop.call_soon_threadsafe(future.set_result, (status, res, err))

    def connect_signals(task: ThreadTask):
        task.signals.task_error.connect(lambda err: set_future_result(FinalTaskStatus.ERROR, err=err))
        task.signals.task_successful.connect(lambda res: set_future_result(FinalTaskStatus.SUCCESS, res=res))
        task.signals.task_cancelled.connect(lambda: set_future_result(FinalTaskStatus.CANCELLED))

    async def execute_wait() -> tuple[FinalTaskStatus, Any, Any]:
        return await future

    return connect_signals, execute_wait

async def execute_command_string(context: ApplicationContext, command_str: str) -> CommandResponse:
    """Process a console command from a string asynchronously. Should never raise."""
    try:
        command_id, arguments = split_command(command_str)
    except ValueError as err:
        return CommandResponse(f"Invalid command syntax: {err}", MessageType.ERROR)

    if not command_id:
        return CommandResponse("No command provided.", MessageType.ERROR)

    if not command_exists(command_id):
        return _handle_unknown_command(command_id)

    try:
        parsed_arguments = deserialize_command(command_id, arguments)
    except CommandArgumentParseError as err:
        return CommandResponse(f"Invalid arguments: {err}", MessageType.ERROR)
    except CommandParserConfigurationError as err:
        return CommandResponse(f"Internal command configuration error: {err}", MessageType.ERROR)

    if context.task_controller.is_thread_slot_occupied(ThreadTaskSlot.execute_command):
        # Handle the case where the thread slot is already occupied
        return CommandResponse("A command is already being executed.", MessageType.ERROR)

    # 1. Prepare async task waiter to receive the result of _run_command_func_with_args
    connect_signals, execute_wait = create_async_task_waiter()
    command_func = get_command_implementation(command_id)

    context.task_controller.start_task( # Can not raise as slot is checked above
        function=_run_command_func_with_args,
        title=f"Executing command: {command_id}",
        slot=ThreadTaskSlot.execute_command,
        provide_progress_controller=False,
        pos_args=[context, command_id, command_func, parsed_arguments],
        kw_args={},
        before_start_callback=connect_signals,
    )

    # 3. Await completion via qasync without blocking the Qt event loop
    status, result, error = await execute_wait()
    match status:
        case FinalTaskStatus.SUCCESS:
            return result
        case FinalTaskStatus.ERROR:
            return CommandResponse(f"Command execution failed: {error}", MessageType.ERROR)
        case FinalTaskStatus.CANCELLED:
            return CommandResponse("Command execution was cancelled.", MessageType.ERROR)

def _run_command_func_with_args(
    context: ApplicationContext,
    command_id: str,
    command_func: CommandImplementation,
    parsed_arguments: list[str|int|float|bool],
) -> CommandResponse:
    try:
        result = command_func(context, *parsed_arguments)
        if asyncio.iscoroutine(result):
            response = asyncio.run(result)
        else:
            response = result

    except TypeError as error:
        print(">>> UNEXPECTED ERROR IN COMMAND FUNCTION <<<")
        traceback.print_exc()
        if _is_wrong_argument_count_error(error):
            response = CommandResponse(
                f"Internal Error: registered and implemented arguments of command {_single_quotes(command_id)} do not match: {error}",
                MessageType.ERROR
            )
        else:
            raise

    except Exception as error:
        print(">>> UNEXPECTED ERROR IN COMMAND FUNCTION <<<")
        traceback.print_exc()
        response = CommandResponse(f"Unexpected error executing command {_single_quotes(command_id)}: {error}", MessageType.ERROR)
        
    return response

def _is_wrong_argument_count_error(error: TypeError) -> bool:
    if ("expected at most" in str(error)) or ("expected at least" in str(error)):
        return True

    pattern = re.compile(
        r"takes\s+(\d+)\s+positional\s+argument(?:s)?\s+but\s+(\d+)\s+were\s+given"
    )
    match = pattern.search(str(error))
    return bool(match)

def _handle_unknown_command(command_id: str) -> CommandResponse:
    # When input is close to a known command or alias, show a hint
    # instead of a plain "unknown command" message.
    closest_command = _get_closest_command_name(command_id)
    if closest_command:
        if closest_command in _command_aliases:
            target_command = _command_aliases[closest_command]
            return CommandResponse(
                (
                    f"Unknown command {_single_quotes(command_id)}. "
                    f"Did you mean {_single_quotes(closest_command)} "
                    f"(alias of {_single_quotes(target_command)})?"
                ), MessageType.ERROR
            )
        else:
            return CommandResponse(
                f"Unknown command {_single_quotes(command_id)}. Did you mean {_single_quotes(closest_command)}?",
                MessageType.ERROR
            )
    else:
        return CommandResponse(
            f"Unknown command {_single_quotes(command_id)} (run 'link.help' for more info).",
            MessageType.ERROR
        )

def serialize_command(command_list: list[str|int|float|bool]) -> str:
    """Convert a list of command segments into a single string, quoting properly as necessary."""
    # The console is not cmd.exe, so use Windows argument quoting without cmd shell rules.
    result = " ".join(mslex.quote(str(arg), for_cmd=False) for arg in command_list)
    return result

def split_command(command_string: str) -> tuple[str|None, list[str]]:
    """
    Split a command string into the command ID and its arguments, respecting quotes.
    Raises:
        ValueError: When the command string is malformed (e.g., unbalanced quotes)
    """
    # The console is not cmd.exe, so use Windows argument parsing without cmd shell rules.
    segments = mslex.split(command_string, like_cmd=False)
    if len(segments) >= 1:
        return segments[0], segments[1:]
    return None, []

def deserialize_command(command_id: str, argument_values: list[str]) -> list[str|int|float|bool]:
    """
    Convert a split command into a list of typed arguments based on the command's argument specifications.
    Given command should exists.
    Raises:
        ValueError: When given invalid arguments
    """
    return deserialize_command_arguments(
        command_id,
        command_description=get_command_description(command_id),
        argument_values=argument_values,
        argument_specs=get_command_arg_specs(command_id),
    )

def _single_quotes(text: str) -> str:
    """Wraps a string in single quotes."""
    return f"'{text.replace('\'', '\\\'')}'"

def _get_closest_command_name(command_id: str) -> str | None:
    """Find the closest command ID or alias to a user-provided command."""
    candidates = get_all_command_ids() + get_all_command_aliases()
    matches = get_close_matches(command_id, candidates, n=1, cutoff=0.3)
    return matches[0] if matches else None
