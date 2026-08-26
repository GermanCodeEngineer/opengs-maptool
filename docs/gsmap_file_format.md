# GSMAP File Format

The `.gsmap` format is used to store all data associated with an **OpenGS Maptool** project.

A GSMAP file is simply a ZIP archive that follows a predefined directory structure.

This format allows maps to be created, generated, and modified across multiple editing sessions, while also making project sharing easier.

## Structure

A GSMAP file is organized as follows:

```txt
project.json
|_ data/
|_ images/
|_ metadata/
```

### 1. Project Description File

#### File Description

The archive contains a `project.json` file that stores general information about the project.

The following table lists the available properties:

| Property         | Optional | Type             | Description                                                            |
|------------------|----------|------------------|------------------------------------------------------------------------|
| `name`           | No       | `string`         | The project name.                                                      |
| `editor_version` | No       | `string`         | The version of OpenGS Maptool used to create or last edit the project. |
| `description`    | Yes      | `string \| null` | A short description of the project.                                    |
| `author`         | Yes      | `string \| null` | The project author's name.                                             |

> **Note:** Optional properties may either be omitted entirely or set to `null` when no value is provided.

#### Example

```json
{
    "name": "My Awesome Project",
    "editor_version": "0.4",
    "description": null,
    "author": "John Doe"
}
```

### 2. Data Directory

This directory contains all JSON data files associated with the project.

For example, if territories have been generated, this directory will contain a file named `territory_data.json`.

> For more information about data file formats, see the [Data Export Format](data_export_format.md) documentation page.

### 3. Images Directory

This directory contains all images used by the project.

Each generated map layer (land, boundaries, territories, etc.) is stored here in PNG format.

> For more information about image file formats, see the [Image Export Format](image_export_format.md) documentation page.

### 4. Metadata Directory

This directory contains files required by the editor to perform various operations.

For example, metadata files are required for **province** generation and other editor-specific features.

## Additional Information

The GSMAP format may evolve between OpenGS Maptool releases.

Although automatic migration tools will be provided for compatible versions whenever possible, it is strongly recommended to keep regular backups of your projects before upgrading.
