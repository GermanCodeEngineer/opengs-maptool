"""
Debug script to check coordinate ranges and image dimensions.
"""
import json
from pathlib import Path
from PIL import Image


output_dir = Path(__file__).parent / "output"

# Load images
cont_areas_image = Image.open(output_dir / "contareasimage.png")
territory_image = Image.open(output_dir / "territoryimage.png")
province_image = Image.open(output_dir / "provinceimage.png")

print("Image dimensions:")
print(f"  Continuous areas: {cont_areas_image.size}")
print(f"  Territories: {territory_image.size}")
print(f"  Provinces: {province_image.size}")

# Load data
cont_areas_data = json.loads((output_dir / "contareasdata.json").read_text())
territory_data = json.loads((output_dir / "territorydata.json").read_text())
province_data = json.loads((output_dir / "provincedata.json").read_text())

def check_coords(name, data, img_size):
    print(f"\n{name}:")
    if not data:
        print("  No data!")
        return
    
    sample = data[0]
    print(f"  Sample entry: {sample}")
    
    # Check for coordinate fields
    has_local = "local_x" in sample and "local_y" in sample
    has_global = "global_x" in sample and "global_y" in sample
    
    print(f"  Has local_x/y: {has_local}")
    print(f"  Has global_x/y: {has_global}")
    
    # Get coordinate ranges
    if has_global:
        coords = [(d.get("global_x"), d.get("global_y")) for d in data]
    elif has_local:
        coords = [(d.get("local_x"), d.get("local_y")) for d in data]
    else:
        print("  ERROR: No coordinate fields found!")
        return
    
    valid_coords = [(x, y) for x, y in coords if x is not None and y is not None]
    
    if valid_coords:
        xs = [x for x, y in valid_coords]
        ys = [y for x, y in valid_coords]
        print(f"  X range: {min(xs):.1f} to {max(xs):.1f}")
        print(f"  Y range: {min(ys):.1f} to {max(ys):.1f}")
        print(f"  Image bounds: (0, 0) to {img_size}")
        
        out_of_bounds = sum(1 for x, y in valid_coords if x < 0 or y < 0 or x >= img_size[0] or y >= img_size[1])
        print(f"  Out of bounds: {out_of_bounds}/{len(valid_coords)}")

check_coords("Continuous Areas", cont_areas_data, cont_areas_image.size)
check_coords("Territories", territory_data, territory_image.size)
check_coords("Provinces", province_data, province_image.size)
