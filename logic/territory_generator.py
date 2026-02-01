import config
import numpy as np
from collections import defaultdict, deque
from PIL import Image
from logic.utils import NumberSeries, is_sea_color

territory_colors: set[tuple[int, int, int]] = set()


def _build_province_adjacency(province_image: Image.Image, province_data: list, boundary_image: Image.Image | None) -> dict:
    """Build adjacency graph of provinces that aren't separated by boundaries."""
    
    p_arr = np.array(province_image, copy=False)
    h, w, _ = p_arr.shape
    
    # Build color to province_id mapping
    color_to_pid = {}
    for p in province_data:
        color_to_pid[(p["R"], p["G"], p["B"])] = p["province_id"]
    
    # Build boundary mask if available
    if boundary_image is not None:
        b_arr = np.array(boundary_image, copy=False)
        if b_arr.ndim == 3:
            r, g, b = config.BOUNDARY_COLOR
            boundary_mask = (
                (b_arr[..., 0] == r) &
                (b_arr[..., 1] == g) &
                (b_arr[..., 2] == b)
            )
        else:
            (val,) = config.BOUNDARY_COLOR[:1]
            boundary_mask = (b_arr == val)
    else:
        boundary_mask = np.zeros((h, w), dtype=bool)
    
    # Find adjacent provinces (not separated by boundaries)
    adjacency = defaultdict(set)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for y in range(h):
        for x in range(w):
            if boundary_mask[y, x]:
                continue
                
            current_color = tuple(p_arr[y, x])
            current_pid = color_to_pid.get(current_color)
            if current_pid is None:
                continue
            
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if boundary_mask[ny, nx]:
                        continue
                    
                    neighbor_color = tuple(p_arr[ny, nx])
                    neighbor_pid = color_to_pid.get(neighbor_color)
                    
                    if neighbor_pid and neighbor_pid != current_pid:
                        adjacency[current_pid].add(neighbor_pid)
    
    return adjacency


def _assign_provinces_to_territories(province_data: list, num_territories: int, adjacency: dict, series: NumberSeries, used_colors: set, ptype: str) -> dict:
    """Group provinces into territories using region growing, respecting boundaries."""
    
    if num_territories <= 0:
        return {}
    
    # Filter provinces by type
    typed_provinces = [p for p in province_data if p["province_type"] == ptype]
    if not typed_provinces:
        return {}
    
    num_territories = min(num_territories, len(typed_provinces))
    
    # Pick seed provinces
    rng = np.random.default_rng(config.RNG_SEED)
    seed_provinces = list(rng.choice(typed_provinces, num_territories, replace=False))
    
    # Initialize territories
    province_to_territory = {}
    territory_metadata = {}
    
    queue = deque()
    for i, seed_prov in enumerate(seed_provinces):
        tid = series.get_id()
        if tid is None:
            continue
        
        r, g, b = color_from_id(i, ptype, used_colors)
        territory_metadata[tid] = {
            "territory_id": tid,
            "territory_type": ptype,
            "R": r, "G": g, "B": b,
            "province_ids": []
        }
        
        province_to_territory[seed_prov["province_id"]] = tid
        territory_metadata[tid]["province_ids"].append(seed_prov["province_id"])
        queue.append(seed_prov["province_id"])
    
    # Grow territories by adding adjacent unassigned provinces
    while queue:
        current_pid = queue.popleft()
        current_tid = province_to_territory[current_pid]
        
        # Check all adjacent provinces
        for neighbor_pid in adjacency.get(current_pid, []):
            if neighbor_pid not in province_to_territory:
                # Find the province data
                neighbor_prov = next((p for p in province_data if p["province_id"] == neighbor_pid), None)
                if neighbor_prov and neighbor_prov["province_type"] == ptype:
                    province_to_territory[neighbor_pid] = current_tid
                    territory_metadata[current_tid]["province_ids"].append(neighbor_pid)
                    queue.append(neighbor_pid)
    
    # Calculate territory centers (most central province)
    for tid, terr_data in territory_metadata.items():
        province_ids = terr_data["province_ids"]
        if not province_ids:
            continue
        
        # Get all provinces in this territory
        terr_provinces = [p for p in province_data if p["province_id"] in province_ids]
        
        # Calculate centroid of all provinces
        avg_x = sum(p["x"] for p in terr_provinces) / len(terr_provinces)
        avg_y = sum(p["y"] for p in terr_provinces) / len(terr_provinces)
        
        # Find the province closest to the centroid
        closest_prov = min(
            terr_provinces,
            key=lambda p: (p["x"] - avg_x) ** 2 + (p["y"] - avg_y) ** 2
        )
        
        # Store the center coordinates
        terr_data["x"] = closest_prov["x"]
        terr_data["y"] = closest_prov["y"]
    
    return territory_metadata


def color_from_id(index: int, ptype: str, used_colors: set) -> tuple[int, int, int]:
    rng = np.random.default_rng(index + 1)

    while True:
        if ptype == "ocean":
            r = rng.integers(0, 60)
            g = rng.integers(0, 80)
            b = rng.integers(100, 180)
        else:
            r, g, b = map(int, rng.integers(0, 256, 3))

        color = (int(r), int(g), int(b))
        if color not in used_colors:
            used_colors.add(color)
            return color


def generate_territory_map(main_layout):
    territory_colors.clear()
    main_layout.progress.setVisible(True)
    main_layout.progress.setValue(10)

    boundary_image = main_layout.boundary_image_display.get_image()
    land_image = main_layout.land_image_display.get_image()

    if boundary_image is None and land_image is None:
        raise ValueError(
            "Need at least boundary OR ocean image to determine map size.")

    # BOUNDARY MASK
    if boundary_image is not None:
        b_arr = np.array(boundary_image, copy=False)

        if b_arr.ndim == 3:
            r, g, b = config.BOUNDARY_COLOR
            boundary_mask = (
                (b_arr[..., 0] == r) &
                (b_arr[..., 1] == g) &
                (b_arr[..., 2] == b)
            )
        else:
            (val,) = config.BOUNDARY_COLOR[:1]
            boundary_mask = (b_arr == val)

        map_h, map_w = boundary_mask.shape

    else:
        boundary_mask = None

    # LAND / SEA MASKS
    if land_image is not None:
        o_arr = np.array(land_image, copy=False)
        sea_mask = is_sea_color(o_arr)
        land_mask = ~sea_mask

        if boundary_mask is None:
            map_h, map_w = sea_mask.shape
    else:
        if boundary_mask is None:
            raise ValueError("Could not determine map size.")

        sea_mask = np.zeros((map_h, map_w), dtype=bool)
        land_mask = np.ones((map_h, map_w), dtype=bool)

    # NUMBER SERIES FOR TERRITORIES
    series = NumberSeries(
        config.TERRITORY_ID_PREFIX,
        config.TERRITORY_ID_START,
        config.TERRITORY_ID_END
    )

    # GENERATE TERRITORIES by grouping provinces
    land_points = main_layout.territory_land_slider.value()
    sea_points = main_layout.territory_ocean_slider.value()

    province_image = main_layout.province_image_display.get_image()
    province_data = main_layout.province_data

    # Build province adjacency graph (respecting boundaries)
    adjacency = _build_province_adjacency(province_image, province_data, boundary_image)

    main_layout.progress.setValue(30)

    # Generate territories by grouping provinces
    land_territories = _assign_provinces_to_territories(
        province_data, land_points, adjacency, series, territory_colors, "land"
    )

    main_layout.progress.setValue(50)

    sea_territories = _assign_provinces_to_territories(
        province_data, sea_points, adjacency, series, territory_colors, "ocean"
    )

    # Combine metadata
    metadata = list(land_territories.values()) + list(sea_territories.values())

    # Build terrain_province_map
    terrain_province_map = {}
    for terr in metadata:
        terrain_province_map[terr["territory_id"]] = terr["province_ids"]

    # Build province-based territory image
    territory_province_image = build_province_based_territory_image(
        province_image,
        province_data,
        metadata
    )

    # Display THIS instead of the raw territory map
    main_layout.territory_image_display.set_image(territory_province_image)
    main_layout.territory_data = metadata

    main_layout.progress.setValue(100)
    main_layout.terrain_province_map = terrain_province_map

    # print(terrain_province_map)
    main_layout.button_exp_terr_img.setEnabled(True)
    main_layout.button_exp_terr_csv.setEnabled(True)
    main_layout.button_exp_terr_json.setEnabled(True)
    return territory_province_image, metadata


def build_province_based_territory_image(province_image, province_data, territory_data):

    p_arr = np.array(province_image, copy=False)
    h, w, _ = p_arr.shape

    # Build lookup: province_id -> territory color
    province_to_territory_color = {}

    for terr in territory_data:
        tcolor = (terr["R"], terr["G"], terr["B"])
        for pid in terr["province_ids"]:
            province_to_territory_color[pid] = tcolor

    # Build lookup: province color -> province_id
    color_to_pid = {}
    for p in province_data:
        color_to_pid[(p["R"], p["G"], p["B"])] = p["province_id"]

    out = np.zeros((h, w, 3), np.uint8)

    for y in range(h):
        for x in range(w):
            rgb = tuple(p_arr[y, x])
            pid = color_to_pid.get(rgb)
            if pid is None:
                continue

            terr_color = province_to_territory_color.get(pid)
            if terr_color is None:
                continue

            out[y, x] = terr_color

    return Image.fromarray(out)
