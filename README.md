<img width="350" height="350" alt="ogs-mt-logo" src="https://github.com/user-attachments/assets/d03854c8-c2e1-468f-9f8a-269f498d169c" />

# Open Grand Strategy - Map Tool 
The OpenGS Map Tool is a specialized utility designed to streamline the creation of map data for use in grand strategy games. 
Province and territory maps form the backbone of these games, defining the geographical regions that players interact with.

## Features
- Generate and Export province maps
- Generate and Export province data
- Generate and Export territory maps
- Generate and Export territory data

## Showcase
<img width="2200" height="2318" alt="image" src="https://github.com/user-attachments/assets/1ad0250a-0a50-4bbd-b616-0e215a7ed2bc" />
<img width="2200" height="2308" alt="image" src="https://github.com/user-attachments/assets/7afe9e4c-648d-4e63-9636-a8df47bfba27" />

## How to install
### Option 1 (Windows only):
1. "Releases" section in Github
2. Download and unpack "ogs_maptool.zip"
3. Run the Executable

### Option 2:
1. Clone the repository
2. Install the package on your device (Python 3.12+ required) by running:
	```sh
	pip install .
	```
3. Start project by running "python main.py -gui"

## Terms Definition

- **Area**: A country or a separate island. The largest continuous region, not split by boundaries. Each area can contain multiple density samples.
- **Density Sample**: A subdivision of an area, where density is averaged. Used to control the number of regions in different parts of the map. Children of areas.
- **Territory**: A subdivision of a density sample. Larger than a province, used for coarse regional division.
- **Province**: A subdivision of a territory. The smallest region, used for fine-grained control.

## Local vs. Global Fields

- **Global fields** (e.g., `global_bbox`, `global_center`): Coordinates or bounding boxes in the context of the entire map image.
- **Local fields** (e.g., `local_bbox`, `local_center`): Coordinates or bounding boxes relative to the parent region (area, density sample, or territory).

When cropping or processing a region, use the global bounding box to crop from the full image. Local bounding boxes are useful for operations within a parent region.

## Performance Tips

- Splitting complex or weird-shaped areas (especially large oceans) into smaller regions improves performance and accuracy.
- Use density/border image to control region sizes: higher density = more, smaller regions; lower density = fewer, larger regions.

## How to Create Classification and Boundary/Density Images

1. **Boundary Image**: Should have pure black lines (RGB 0,0,0) for boundaries. The greyscale of all other values can be used to encode density multipliers (0 = 4x fewer regions, 255 = 4x more regions). <br> Hint: Avoid creating islands or regions that are only borders (i.e., surrounded entirely by black pixels), as these may not be processed correctly.
2. **Classification Image**: Should use RGB (5, 20, 18) for ocean, (150, 68, 192) for land and (0, 255, 0) for lakes.
3. Always use the same resolution for boundary/density and classification images to avoid errors and misalignment.

## GUI Usage

The graphical interface provides a step-by-step workflow:

1. **Create Density Image**: Import and clean your boundary image. Adjust density using grayscale values.
2. **Input Images**: Import your final boundary and classification images.
3. **Generate Areas**: Generate continuous areas (countries/islands) from your images.
4. **Process Density in Areas**: Calculate density samples from areas.
5. **Generate Territories**: Subdivide density samples into territories.
6. **Generate Provinces**: Subdivide territories into provinces.

You can export maps and data (CSV/JSON) at each step.

## Python Usage

Use the `ProcessMapTool` class for programmatic map generation. See the code for up-to-date usage examples.

## Best Practices

- Always split very large or complex regions (like world oceans) for better performance.
- Use density multipliers to control region sizes.
- Ensure all input images are the same size.

## Contributions
Contributions can come in many forms and all are appreciated:
- Feedback
- Code improvements
- Added functionality

## Discord 
Follow and/or support the project on [OpenGS Discord Server](https://discord.gg/6apDaJrY)

## Delivered and maintained by 
<img width="350" height="350" alt="gsi-logo" src="https://github.com/user-attachments/assets/e7210566-7997-4d82-845e-48f249d439a0" />
