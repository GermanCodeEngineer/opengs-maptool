import opengs_maptool.config as config

import webbrowser

def open_github() -> None:
    webbrowser.open(config.GITHUB_URL)

def open_console_help() -> None:
    webbrowser.open(config.CONSOLE_HELP_URL)

def open_discord() -> None:
    webbrowser.open(config.DISCORD_URL)
