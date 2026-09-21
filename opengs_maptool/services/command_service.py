from __future__ import annotations

# Import package to trigger decorator registrations
import opengs_maptool.services.commands  # noqa: F401

# Re-export only the public functions required by external modules
from opengs_maptool.services.command_core import (
    execute_command_string,
    serialize_command,
    _single_quotes,
)

def wrap_in_single_quotes(value: str) -> str:
    return _single_quotes(value)

__all__ = [
    "execute_command_string",
    "serialize_command",
    "wrap_in_single_quotes",
]
