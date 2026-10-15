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
| `[x]` | `project.name.set` | `<name>` | Changes the name in the project details |
| `[x]` | `project.description.set` | `<description>` | Changes the description in the project details |
| `[x]` | `project.author.set` | `<author>` | Changes the author in the project details |
||||
| `[x]` | `land.image.import` | `<path>` | Imports the land image from a file |
| `[x]` | `land.color_settings.set` | `<land_type: land\|ocean\|lake> <color>` | Changes the colors expected in the land image for land, oceans and lakes |
| `[x]` | `boundary.image.import` | `<path>` | Imports the boundary image from a file |
| `[x]` | `density.image.import` | `<path>` | Imports the density image from a file |
| `[x]` | `terrain.image.import` | `<path>` | Imports the terrain image from a file |
||||
| `[x]` | `density.image.remove` | None | Removes the current density image |
| `[x]` | `density.image.normalize` | None | Normalizes density values |
| `[x]` | `density.image.equator_distribute` | None | Generates equator-based density distribution |
| `[x]` | `density.exclude_ocean_territories` | `<activate: true\|false>` | Sets whether density applies to ocean territories |
| `[x]` | `density.exclude_ocean_provinces` | `<activate: true\|false>` | Sets whether density applies to ocean provinces |
||||
| `[x]` | `territory.image.generate` | None | Generates the territory map |
| `[x]` | `territory.image.export` | `<path>` | Exports the territory image |
| `[x]` | `territory.definitions.export` | `<path>` | Exports territory definitions |
| `[x]` | `territory.history.export` | `<path> <format>` | Exports territory history |
| `[x]` | `territory.land_density.set` | `<value>` | Sets land territory density |
| `[x]` | `territory.oceanic_density.set` | `<value>` | Sets oceanic territory density |
| `[x]` | `territory.density_strength.set` | `<factor>` | Sets territory density strength |
| `[x]` | `territory.jagged_land.set` | `<strength>` | Sets how strongly jagged the land borders of territories should be |
| `[x]` | `territory.jagged_oceanic.set` | `<strength>` | Sets how strongly jagged the oceanic borders of territories should be |
||||
| `[x]` | `province.image.generate` | None | Generates the province map |
| `[x]` | `province.image.export` | `<path>` | Exports the province image |
| `[x]` | `province.definitions.export` | `<path> <format>` | Exports province definitions |
| `[x]` | `province.land_density.set` | `<value>` | Sets land province density |
| `[x]` | `province.oceanic_density.set` | `<value>` | Sets oceanic province density |
| `[x]` | `province.density_strength.set` | `<factor>` | Sets province density strength |
| `[x]` | `province.jagged_land.set` | `<strength>` | Sets how strongly jagged the land borders of provinces should be |
| `[x]` | `province.jagged_oceanic.set` | `<strength>` | Sets how strongly jagged the oceanic borders of provinces should be |
||||
| `[x]` | `wait.tasks.completed` | None | Waits until all background tasks are completed |

## Aliases
| Done | Alias Name | Real Command |
|----|---------|-----------|
| `[x]` | `?` | `link.help` |
| `[x]` | `h` | `link.help` |
| `[x]` | `help` | `link.help` |
