import config
import numpy as np
from numpy.typing import NDArray
from typing import Any
from PIL import Image
from gceutils import grepr_dataclass
from logic.boundaries_to_cont import convert_boundaries_to_cont_areas, assign_borders_to_areas, classify_pixels_by_color
from logic.cont_to_regions import convert_all_cont_areas_to_regions
from logic.utils import NumberSeries


@grepr_dataclass(validate=False, frozen=True)
class MapToolResult:
    """
    Dataclass Containing Results of the Map Tool
    - territory and province maps
    - continous areas map
    - cleaned land/ocean/lake classification
    - data of continous areas, territories & provinces 
    """
    cont_areas_image: Image.Image
    cont_areas_data: list[dict[str, Any]]
    class_image: Image.Image
    class_counts: dict[str, int]
    territory_image: Image.Image
    territory_data: list[dict[str, Any]]
    province_image: Image.Image
    province_data: list[dict[str, Any]]

@grepr_dataclass(init=False)
class MapTool:
    """
    Open Grand Strategy Map Tool, which can be directly used in python.
    """
    land_image: NDArray[np.uint8]
    boundary_image: NDArray[np.uint8]
    pixels_per_land_territory: int
    pixels_per_water_territory: int
    pixels_per_land_province: int
    pixels_per_water_province: int
    lloyd_iterations: int

    def __init__(self,
            land_image: Image.Image,
            boundary_image: Image.Image,
            pixels_per_land_territory: int = config.PIXELS_PER_LAND_TERRITORY_DEFAULT,
            pixels_per_water_territory: int = config.PIXELS_PER_WATER_TERRITORY_DEFAULT,
            pixels_per_land_province: int = config.PIXELS_PER_LAND_PROVINCE_DEFAULT,
            pixels_per_water_province: int = config.PIXELS_PER_WATER_PROVINCE_DEFAULT, # 1/5th
            lloyd_iterations: int = 2,
        ) -> None:
        """
        Initialize MapTool with input images and parameters.
        
        Args:
            land_image: PIL Image containing land/ocean/lake classification
            boundary_image: PIL Image containing (country) boundaries and density information(in blue channel)
            pixels_per_land_territory: Approximate pixels per land territory
            pixels_per_water_territory: Approximate pixels per water territory
            pixels_per_land_province: Approximate pixels per land province
            pixels_per_water_province: Approximate pixels per water province
            lloyd_iterations: Number of Lloyd's algorithm iterations for province and territory generation 
        """
        super().__init__()

        self.land_image = np.array(land_image.convert("RGBA"))
        self.boundary_image = np.array(boundary_image.convert("RGBA"))
        self.pixels_per_land_territory = pixels_per_land_territory
        self.pixels_per_water_territory = pixels_per_water_territory
        self.pixels_per_land_province = pixels_per_land_province
        self.pixels_per_water_province = pixels_per_water_province
        self.lloyd_iterations = lloyd_iterations
    
   
    def generate(self) -> MapToolResult:
        """
        Generate province and territory maps from stored input images.
        Calls event listener methods on completing a map.
        
        This method orchestrates the full map generation pipeline:
        1. Converts boundaries to continuous areas
        2. Classifies pixels by land/water type
        3. Generates territories from continuous areas
        4. Generates provinces from territories
        """
        cont_areas_image, cont_areas_image_buffer, cont_areas_data = self._generate_cont_areas()
        class_image, class_image_buffer, class_counts = self._generate_type_classification()
        territory_image, territory_image_buffer, territory_data = self._generate_territories(cont_areas_image_buffer, cont_areas_data, class_image_buffer, class_counts)
        province_image, province_image_buffer, province_data = self._generate_provinces(territory_image_buffer, territory_data, class_image_buffer, class_counts)
        return MapToolResult(
            cont_areas_image, cont_areas_data,
            class_image, class_counts,
            territory_image, territory_data,
            province_image, province_data,
        )
    
    def _generate_cont_areas(self) -> tuple[Image.Image, NDArray[np.uint8], list[dict[str, Any]]]:
        areas_with_borders_image, cont_areas_data = convert_boundaries_to_cont_areas(
            self.boundary_image, 
            config.CONT_AREAS_RNG_SEED,
            min_area_pixels=50  # Filter out tiny areas & islands
        )
        cont_areas_image = assign_borders_to_areas(areas_with_borders_image)
        args = (Image.fromarray(cont_areas_image), cont_areas_image, cont_areas_data)
        if callable(getattr(self, "on_cont_areas_generated", None)):
            self.on_cont_areas_generated(*args)
        return args
    
    def _generate_type_classification(self) -> tuple[Image.Image, NDArray[np.uint8], dict[str, int]]:
        class_image, class_counts = classify_pixels_by_color(np.array(self.land_image), export_colors=True)
        args = (Image.fromarray(class_image), class_image, class_counts)
        if callable(getattr(self, "on_type_classification_generated", None)):
            self.on_type_classification_generated(*args)
        return args
    
    def _generate_territories(self,
        cont_areas_image: NDArray[np.uint8], cont_areas_data: list[dict[str, Any]],
        class_image: NDArray[np.uint8], class_counts: dict[str, int],
    ) -> tuple[Image.Image, NDArray[np.uint8], list[dict[str, Any]]]:
        territory_image, territory_data = convert_all_cont_areas_to_regions(
            cont_areas_image=cont_areas_image,
            cont_areas_metadata=cont_areas_data,
            class_image=class_image,
            class_counts=class_counts,
            pixels_per_land_region=self.pixels_per_land_territory,
            pixels_per_water_region=self.pixels_per_water_territory,
            fn_new_number_series=lambda area_meta: NumberSeries(
                f"{area_meta['region_id']}-{config.TERRITORY_ID_PREFIX}", config.SERIES_ID_START, config.SERIES_ID_END
            ),
            rng_seed=config.TERRITORIES_RNG_SEED,
            lloyd_iterations=self.lloyd_iterations,
            tqdm_description="Generating territories from areas",
            tqdm_unit="areas"
        )

        # Replace ids with correct format
        number_series = NumberSeries(config.TERRITORY_ID_PREFIX, config.SERIES_ID_START, config.SERIES_ID_END)
        for territory in territory_data:
            territory["region_id"] = number_series.get_id()
        
        args = (Image.fromarray(territory_image), territory_image, territory_data)
        if callable(getattr(self, "on_territories_generated", None)):
            self.on_territories_generated(*args)
        return args

    def _generate_provinces(self,
        territory_image: NDArray[np.uint8], territory_data: list[dict[str, Any]],
        class_image: NDArray[np.uint8], class_counts: dict[str, int],
    ) -> tuple[Image.Image, NDArray[np.uint8], list[dict[str, Any]]]:
        province_image, province_data = convert_all_cont_areas_to_regions(
            cont_areas_image=territory_image,
            cont_areas_metadata=territory_data,
            class_image=class_image,
            class_counts=class_counts,
            pixels_per_land_region=self.pixels_per_land_province,
            pixels_per_water_region=self.pixels_per_water_province,
            fn_new_number_series=lambda territory_meta: NumberSeries(
                f"{territory_meta['region_id']}-{config.PROVINCE_ID_PREFIX}", config.SERIES_ID_START, config.SERIES_ID_END
            ),
            rng_seed=config.PROVINCES_RNG_SEED,
            lloyd_iterations=self.lloyd_iterations,
            tqdm_description="Generating provinces from territories",
            tqdm_unit="territories",
        )

        args = (Image.fromarray(province_image), province_image, province_data)
        if callable(getattr(self, "on_provinces_generated", None)):
            self.on_provinces_generated(*args)
        return args


    def on_cont_areas_generated(self,
        cont_areas_image: Image.Image, cont_areas_image_buffer: NDArray[np.uint8], cont_areas_data: list[dict[str, Any]]) -> None: ...
    def on_type_classification_generated(self,
        class_image: Image.Image, class_image_buffer: NDArray[np.uint8], class_counts: dict[str, int]) -> None: ...
    def on_territories_generated(self,
        territory_image: Image.Image, territory_image_buffer: NDArray[np.uint8], territory_data: list[dict[str, Any]]) -> None: ...
    def on_provinces_generated(self,
        province_image: Image.Image, province_image_buffer: NDArray[np.uint8], province_data: list[dict[str, Any]]) -> None: ...
