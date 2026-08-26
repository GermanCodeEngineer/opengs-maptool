# The Console

The console is a tool located in the right-hand panel. It allows you to execute various commands to retrieve information or perform actions.
This document explains the console, [the used command syntax](#command-syntax) and has [a list of commands](#list-of-commands).
Using the two buttons located below the console, you can export or import a console's history into the editor.

> Please keep in mind that this feature is experimental and still in its early stages.
<br>
> Additionally, in the future, the console will contain logs of all actions performed in the editor.

## Command Syntax

The console command structure follows standard command-line and [Python `argparse` patterns](https://docs.python.org/3/howto/argparse.html). A command consists of a unique dot-separated command identifier, followed by **positional arguments** and/or **optional flags**:

```text
command.name <required_positional_arguments> [--optional_flag] [--optional_setting value]
```

### 1. Required Positional Arguments

Required inputs have no dashes and must be provided in the exact order specified by the command:

```text
project.open "my-project.gsmap"
land.image.import "C:\Maps\land_layer.png"
```

* **Quoting:** If an argument value contains spaces, backslashes or other special characters (such as file paths), it is safter and sometimes necessary to enclose it in double quotes (`"..."`). Single quotes are **not** supported (`'...'`).

### 2. Optional Arguments (Flags & Settings)

Options modify the behavior of a command. They are identified by double dashes (`--`) and can be placed anywhere after the command identifier:

* **Boolean Switches (Flags):** Options like `--force` act as on/off switches. You do not need to provide a value; passing the flag turns it on.
* **Value Settings:** Options that require data are provided using either a space or an equals sign.

#### Example
A **hypothetical** command declared as `example.export <path> [--format FORMAT] [--force]` could be used like this:
```text
project.new --force
project.open "my-project.opengs" --force
example.export "output/map.png" --format png --scale 2
example.export "output/map.png" --scale=2 --format=png
```

---

## Interactive Console Features

### Dynamic Help Menus

Every command has a built-in help manual. Appending `-h` or `--help` to any command will display its formal syntax and detailed argument descriptions directly in your console:

```text
project.open --help
```

### Smart Command Suggestions

If you mistype a command name, the console automatically scans for a close matching command or alias and suggest the correct syntax:

```text
> projct.open "map.gsmap"
Unknown command 'projct.open'. Did you mean 'project.open'?
```




# List of Commands
Here is a list of the commands available in the editor console.

### Legend
- Not done: `[]`
- In Progress `[.]`
- Can be improved/Todos: `[t]`
- Done: `[x]`

### Planned Commands
| Done | Command | Arguments | Description |
|----|---------|-----------|-------------|
| `[x]` | `link.help` | None | Displays the link for the command documentation
| `[x]` | `link.discord` | None | Displays the link to the OpenGS Discord server |
| `[x]` | `link.github` | None | Displays the link to the editor's GitHub repository |
||||
| `[x]` | `console.commands.list` | None | Display a simplified list of commands |
| `[x]` | `console.history.clear` | None | Clears the console message history |
| `[x]` | `console.history.list` | None | Displays the history of console commands entered by the user or GUI |
||||
| `[x]` | `project.new` | None | Creates a new project |
| `[x]` | `project.open` | `<path>` | Opens an existing project file |
| `[x]` | `project.save` | `[--path PATH]` | Saves the current project to the already set path if it is set |
| `[]` | `project.details` | `[...]` | **With subcommands** to read or edit author, description, etc. |
||||
| `[x]` | `land.image.import` | `<path>` | Imports the land image from a file |
| `[x]` | `boundary.image.import` | `<path>` | Imports the boundary image from a file |
| `[x]` | `density.image.import` | `<path>` | Imports the density image from a file |
| `[x]` | `terrain.image.import` | `<path>` | Imports the terrain image from a file |
||||
| `[.]` | `density.image.remove` | None | Removes the current density image |
| `[.]` | `density.image.normalize` | None | Normalizes density values |
| `[.]` | `density.image.equator_distribute` | None | Generates equator-based density distribution |
||||
| `[]` | `territory.image.generate` | None | Generates the territory map |
| `[]` | `territory.image.export` | `<path>` | Exports the territory image |
| `[]` | `territory.definitions.export` | `<path> <format>` | Exports territory definitions |
| `[]` | `territory.history.export` | `<path> <format>` | Exports territory history |
| `[]` | `territory.exclude_ocean` | `<true\|false>` | Sets whether territory generation excludes ocean |
| `[]` | `territory.land_density` | `<value>` | Sets land territory density |
| `[]` | `territory.ocean_density` | `<value>` | Sets ocean territory density |
| `[]` | `territory.density_strength` | `<value>` | Sets territory density strength |
| `[]` | `territory.jagged_land` | `<true\|false>` | Sets jagged land border behavior for territories |
| `[]` | `territory.jagged_ocean` | `<true\|false>` | Sets jagged ocean border behavior for territories |
||||
| `[]` | `province.image.generate` | None | Generates the province map |
| `[]` | `province.image.export` | `<path>` | Exports the province image |
| `[]` | `province.definitions.export` | `<path> <format>` | Exports province definitions |
| `[]` | `province.exclude_ocean` | `<true\|false>` | Sets whether province generation excludes ocean |
| `[]` | `province.land_density` | `<value>` | Sets land province density |
| `[]` | `province.ocean_density` | `<value>` | Sets ocean province density |
| `[]` | `province.density_strength` | `<value>` | Sets province density strength |
| `[]` | `province.jagged_land` | `<true\|false>` | Sets jagged land border behavior for provinces |
| `[]` | `province.jagged_ocean` | `<true\|false>` | Sets jagged ocean border behavior for provinces |

## Aliases
| Done | Alias Name | Real Command |
|----|---------|-----------|
| `[]` | `?` | `link.help` |
| `[]` | `h` | `link.help` |
| `[]` | `help` | `link.help` |

## More optional future commands
- Commands to export input images (purpose: e.g. pull land image from maptool save file)
- Commands to export json file of project information
