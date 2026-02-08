"""
Visualize the centers (capitals) of continuous areas, territories, and provinces on three maps.

This script can be run standalone to:
1. Generate maps from example inputs using MapTool
2. Save the results (images and data) to the output folder
3. Visualize the region centers on each map
"""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Add parent directory to path so we can import from logic
sys.path.insert(0, str(Path(__file__).parent.parent))

from logic.maptool import MapTool
from logic.export_module import export_to_json, export_to_csv


def generate_maps(output_dir: Path) -> dict:
    """Generate all maps using MapTool with example inputs."""
    example_input_dir = Path(__file__).parent.parent / "example_input"
    
    if not example_input_dir.exists():
        raise FileNotFoundError(f"Example input directory not found: {example_input_dir}")
    
    boundary_image_path = example_input_dir / "bound2_density.png"
    land_image_path = example_input_dir / "land2.png"
    
    if not boundary_image_path.exists() or not land_image_path.exists():
        raise FileNotFoundError(f"Required input files not found in {example_input_dir}")
    
    print("Loading input images...")
    boundary_image = Image.open(boundary_image_path)
    land_image = Image.open(land_image_path)
    
    print("Generating maps with MapTool...")
    maptool = MapTool(
        land_image=land_image,
        boundary_image=boundary_image,
    )
    result = maptool.generate()
    
    print("Saving generated maps and data...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save images
    result.cont_areas_image.save(output_dir / "contareasimage.png")
    result.territory_image.save(output_dir / "territoryimage.png")
    result.province_image.save(output_dir / "provinceimage.png")
    
    # Save data as JSON
    export_to_json(result.cont_areas_data, output_dir / "contareasdata.json")
    export_to_json(result.territory_data, output_dir / "territorydata.json")
    export_to_json(result.province_data, output_dir / "provincedata.json")
    
    # Save data as CSV
    export_to_csv(result.cont_areas_data, output_dir / "contareasdata.csv")
    export_to_csv(result.territory_data, output_dir / "territorydata.csv")
    export_to_csv(result.province_data, output_dir / "provincedata.csv")
    
    return {
        "cont_areas": (result.cont_areas_data, result.cont_areas_image),
        "territories": (result.territory_data, result.territory_image),
        "provinces": (result.province_data, result.province_image),
    }


def load_data(output_dir: Path) -> dict:
    """Load all region data and images, generating them if they don't exist."""
    required_files = [
        "contareasdata.json", "territorydata.json", "provincedata.json",
        "contareasimage.png", "territoryimage.png", "provinceimage.png"
    ]
    
    # Check if all files exist
    if all((output_dir / f).exists() for f in required_files):
        print("Loading existing maps and data...")
        cont_areas_data = json.loads((output_dir / "contareasdata.json").read_text())
        territory_data = json.loads((output_dir / "territorydata.json").read_text())
        province_data = json.loads((output_dir / "provincedata.json").read_text())
        
        cont_areas_image = Image.open(output_dir / "contareasimage.png").convert("RGBA")
        territory_image = Image.open(output_dir / "territoryimage.png").convert("RGBA")
        province_image = Image.open(output_dir / "provinceimage.png").convert("RGBA")
        
        return {
            "cont_areas": (cont_areas_data, cont_areas_image),
            "territories": (territory_data, territory_image),
            "provinces": (province_data, province_image),
        }
    else:
        # Generate maps if any are missing
        return generate_maps(output_dir)


def draw_centers(image: Image.Image, data: list[dict], coord_scale: float = 1.0, 
                 circle_radius: int = 5, outline_color: str = "red", 
                 text: bool = False) -> Image.Image:
    """
    Draw circles at the center coordinates of regions.
    
    Args:
        image: PIL Image to draw on
        data: List of region metadata dicts
        coord_scale: Scale factor for coordinates (useful for zooming)
        circle_radius: Radius of circle markers
        outline_color: Color of circle outline
        text: Whether to label with region IDs
    
    Returns:
        Image with centers drawn
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for region in data:
        # Try global coordinates first, fall back to local
        if "global_x" in region and "global_y" in region:
            x = region["global_x"] * coord_scale
            y = region["global_y"] * coord_scale
        elif "local_x" in region and "local_y" in region:
            x = region["local_x"] * coord_scale
            y = region["local_y"] * coord_scale
        else:
            continue
        
        x = int(round(x))
        y = int(round(y))
        
        # Draw circle
        r = circle_radius
        draw.ellipse([x - r, y - r, x + r, y + r], outline=outline_color, width=2)
        
        # Optionally draw text label
        if text:
            region_id = region.get("region_id", "?")
            draw.text((x + r + 2, y), str(region_id), fill=outline_color)
    
    return img_copy


def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    data = load_data(output_dir)
    
    # Visualize continuous areas
    print("Visualizing continuous areas...")
    cont_areas_data, cont_areas_image = data["cont_areas"]
    cont_areas_vis = draw_centers(
        cont_areas_image, cont_areas_data,
        circle_radius=8, outline_color="lime", text=False
    )
    cont_areas_vis.save(output_dir / "contareascenters.png")
    print(f"  Saved: contareascenters.png ({len(cont_areas_data)} centers)")
    
    # Visualize territories
    print("Visualizing territories...")
    territory_data, territory_image = data["territories"]
    territory_vis = draw_centers(
        territory_image, territory_data,
        circle_radius=6, outline_color="cyan", text=False
    )
    territory_vis.save(output_dir / "territorycenters.png")
    print(f"  Saved: territorycenters.png ({len(territory_data)} centers)")
    
    # Visualize provinces
    print("Visualizing provinces...")
    province_data, province_image = data["provinces"]
    province_vis = draw_centers(
        province_image, province_data,
        circle_radius=4, outline_color="yellow", text=False
    )
    province_vis.save(output_dir / "provincecenters.png")
    print(f"  Saved: provincecenters.png ({len(province_data)} centers)")
    
    # Create a combined map with all three overlaid on the province map
    print("Creating combined visualization...")
    combined = province_image.copy()
    
    # Draw territories first (larger circles, darker)
    combined = draw_centers(
        combined, territory_data,
        circle_radius=5, outline_color="blue", text=False
    )
    
    # Then continuous areas (largest circles)
    combined = draw_centers(
        combined, cont_areas_data,
        circle_radius=7, outline_color="lime", text=False
    )
    
    # Finally provinces (smallest circles on top)
    combined = draw_centers(
        combined, province_data,
        circle_radius=3, outline_color="red", text=False
    )
    
    combined.save(output_dir / "allcenters.png")
    print(f"  Saved: allcenters.png (combined visualization)")
    print("\nColor scheme:")
    print("  Lime   = Continuous areas")
    print("  Blue   = Territories")
    print("  Red    = Provinces")


if __name__ == "__main__":
    main()
