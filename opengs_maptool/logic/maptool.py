import numpy as np
from numpy.typing import NDArray
from typing import Any
from PIL import Image
from gceutils import grepr_dataclass
from opengs_maptool.logic.boundaries_to_cont import convert_boundaries_to_cont_areas, assign_borders_to_areas, classify_pixels_by_color, recalculate_bboxes_from_image, classify_continuous_areas, clean_boundary_image
from opengs_maptool.logic.cont_to_regions import convert_all_cont_areas_to_regions
from opengs_maptool.logic.utils import NumberSeries, RegionMetadata
from opengs_maptool import config


@grepr_dataclass(validate=False, frozen=True)
class MapToolResult:
    """
    Dataclass Containing Results of the Map Tool
    - continuous areas map
    - district, territory and province maps
    - data of continuous areas, districts, territories & provinces 
    """
    cont_area_image: Image.Image
    cont_area_data: list[RegionMetadata]
    district_image: Image.Image
    district_data: list[RegionMetadata]
    territory_image: Image.Image
    territory_data: list[RegionMetadata]
    province_image: Image.Image
    province_data: list[RegionMetadata]

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
    cont_areas_rng_seed: int
    districts_rng_seed: int
    territories_rng_seed: int
    provinces_rng_seed: int

    def __init__(self,
            land_image: Image.Image,
            boundary_image: Image.Image,
            pixels_per_land_district: int = config.PIXELS_PER_LAND_DISTRICT_DEFAULT,
            pixels_per_water_district: int = config.PIXELS_PER_WATER_DISTRICT_DEFAULT,
            pixels_per_land_territory: int = config.PIXELS_PER_LAND_TERRITORY_DEFAULT,
            pixels_per_water_territory: int = config.PIXELS_PER_WATER_TERRITORY_DEFAULT,
            pixels_per_land_province: int = config.PIXELS_PER_LAND_PROVINCE_DEFAULT,
            pixels_per_water_province: int = config.PIXELS_PER_WATER_PROVINCE_DEFAULT, # 1/5th
            lloyd_iterations: int = 2,
            cont_areas_rng_seed: int = int(1e6),
            districts_rng_seed: int = int(2e6),
            territories_rng_seed: int = int(3e6),
            provinces_rng_seed: int = int(4e6),
        ) -> None:
        """
        Initialize MapTool with input images and parameters.
        
        Args:
            land_image: **CLEANED** PIL Image containing land/ocean/lake classification (see clean_land_image)
            boundary_image: **CLEANED** PIL Image containing (country) boundaries (see clean_boundary_image)
            pixels_per_land_district: Approximate pixels per land district
            pixels_per_water_district: Approximate pixels per water district
            pixels_per_land_territory: Approximate pixels per land territory
            pixels_per_water_territory: Approximate pixels per water territory
            pixels_per_land_province: Approximate pixels per land province
            pixels_per_water_province: Approximate pixels per water province
            lloyd_iterations: Number of Lloyd's algorithm iterations for province and territory generation 
            cont_areas_rng_seed: RNG seed used for continuous area generation
            districts_rng_seed: RNG seed used for district generation
            territories_rng_seed: RNG seed used for territory generation
            provinces_rng_seed: RNG seed used for province generation
        """
        super().__init__()

        self.land_image = np.array(land_image.convert("RGBA"))
        self.boundary_image = np.array(boundary_image.convert("RGBA"))
        self.pixels_per_land_district = pixels_per_land_district
        self.pixels_per_water_district = pixels_per_water_district
        self.pixels_per_land_territory = pixels_per_land_territory
        self.pixels_per_water_territory = pixels_per_water_territory
        self.pixels_per_land_province = pixels_per_land_province
        self.pixels_per_water_province = pixels_per_water_province
        self.lloyd_iterations = lloyd_iterations
        self.cont_areas_rng_seed = cont_areas_rng_seed
        self.districts_rng_seed = districts_rng_seed
        self.territories_rng_seed = territories_rng_seed
        self.provinces_rng_seed = provinces_rng_seed
    
   
    def generate(self) -> MapToolResult:
        """
        Generate province and territory maps from stored input images.
        Calls event listener methods on completing a map.
        
        This method orchestrates the full map generation pipeline:
        1. Converts boundaries to continuous areas
        2. Generates districts from continuous areas
        3. Generates territories from districts
        4. Generates provinces from territories
        """
        cont_area_image, cont_area_image_buffer, cont_area_data = self._generate_cont_areas()
        
        # Classify continuous areas by land/ocean/lake type
        cont_area_data = classify_continuous_areas(cont_area_image_buffer, np.array(self.land_image), cont_area_data)
        
        district_image, district_image_buffer, district_data = self._generate_districts(cont_area_image_buffer, cont_area_data)
        territory_image, territory_image_buffer, territory_data = self._generate_territories(district_image_buffer, district_data)
        province_image, province_image_buffer, province_data = self._generate_provinces(territory_image_buffer, territory_data)
        return MapToolResult(
            cont_area_image, cont_area_data,
            district_image, district_data,
            territory_image, territory_data,
            province_image, province_data,
        )
    
    def _generate_cont_areas(self, progress_callback=None) -> tuple[Image.Image, NDArray[np.uint8], list[RegionMetadata]]:
        if progress_callback:
            progress_callback(0, 100)

        def boundaries_progress(current, total):
            if progress_callback:
                # Map progress (0-100) to overall progress (0-40)
                progress_callback(int((current / total) * 40), 100)

        areas_with_borders_image, cont_area_data = convert_boundaries_to_cont_areas(
            self.boundary_image,
            self.land_image,
            self.cont_areas_rng_seed,
            min_area_pixels=config.MIN_AREA_PIXELS,  # Filter out tiny areas & islands
            progress_callback=boundaries_progress
        )

        if progress_callback:
            progress_callback(40, 100)

        def border_progress(current, total):
            if progress_callback:
                # Map iteration progress (0-100) to overall progress (40-80)
                progress_callback(40 + int((current / total) * 40), 100)

        cont_area_image = assign_borders_to_areas(areas_with_borders_image, progress_callback=border_progress)

        if progress_callback:
            progress_callback(80, 100)

        def bbox_progress(current, total):
            if progress_callback:
                # Map bbox progress (0-100) to overall progress (80-100)
                progress_callback(80 + int((current / total) * 20), 100)

        # Recalculate bboxes from the final image after border assignment, with progress
        # recalculate_bboxes_from_image does not support progress_callback, so simulate it
        total_regions = len(cont_area_data)
        updated_metadata = []
        for idx, region in enumerate(cont_area_data):
            updated_metadata.append(region)
            if progress_callback and total_regions > 0:
                bbox_progress(idx + 1, total_regions)
        cont_area_data = recalculate_bboxes_from_image(cont_area_image, cont_area_data)

        # Assign proper region_ids (like for territories)
        number_series = NumberSeries(config.AREA_ID_PREFIX, config.SERIES_ID_START, config.SERIES_ID_END)
        for region in cont_area_data:
            region.region_id = number_series.get_id()

        args = (Image.fromarray(cont_area_image), cont_area_image, cont_area_data)
        if callable(getattr(self, "on_cont_areas_generated", None)):
            self.on_cont_areas_generated(*args)
        return args
    
    def _generate_districts(self,
        cont_area_image: NDArray[np.uint8], cont_area_data: list[RegionMetadata],
        progress_callback=None,
    ) -> tuple[Image.Image, NDArray[np.uint8], list[RegionMetadata]]:
        def district_progress(current: int, total: int) -> None:
            if progress_callback:
                # Map progress (0-100) to overall progress (0-90)
                progress_callback(int((current / total) * 90), 100)

        district_image, district_data = convert_all_cont_areas_to_regions(
            cont_area_image=cont_area_image,
            cont_areas_metadata=cont_area_data,
            density_image=self.boundary_image,
            pixels_per_land_region=self.pixels_per_land_district,
            pixels_per_water_region=self.pixels_per_water_district,
            fn_new_number_series=lambda area_meta: NumberSeries(
                f"{area_meta.region_id}-TEMP", config.SERIES_ID_START, config.SERIES_ID_END
            ),
            rng_seed=self.districts_rng_seed,
            lloyd_iterations=self.lloyd_iterations,
            override_density_multiplier=True,
            tqdm_description="Generating districts from areas",
            tqdm_unit="areas",
            progress_callback=district_progress,
        )

        if progress_callback:
            progress_callback(90, 100)

        number_series = NumberSeries(config.DISTRICT_ID_PREFIX, config.SERIES_ID_START, config.SERIES_ID_END)
        for district in district_data:
            district.region_id = number_series.get_id()

        if progress_callback:
            progress_callback(100, 100)

        args = (Image.fromarray(district_image), district_image, district_data)
        if callable(getattr(self, "on_districts_generated", None)):
            self.on_districts_generated(*args)
        return args
    
    def _generate_territories(self,
        district_image: NDArray[np.uint8], district_data: list[RegionMetadata],
        progress_callback=None,
    ) -> tuple[Image.Image, NDArray[np.uint8], list[RegionMetadata]]:
        def territory_progress(current: int, total: int) -> None:
            if progress_callback:
                # Map progress (0-100) to overall progress (0-90)
                progress_callback(int((current / total) * 90), 100)
        
        territory_image, territory_data = convert_all_cont_areas_to_regions(
            cont_area_image=district_image,
            cont_areas_metadata=district_data,
            density_image=self.boundary_image,
            pixels_per_land_region=self.pixels_per_land_territory,
            pixels_per_water_region=self.pixels_per_water_territory,
            fn_new_number_series=lambda area_meta: NumberSeries(
                f"{area_meta.region_id}-TEMP", config.SERIES_ID_START, config.SERIES_ID_END
            ),
            rng_seed=self.territories_rng_seed,
            lloyd_iterations=self.lloyd_iterations,
            override_density_multiplier=False,
            tqdm_description="Generating territories from districts",
            tqdm_unit="districts",
            progress_callback=territory_progress,
        )

        # Replace ids with correct format
        if progress_callback:
            progress_callback(90, 100)
        
        number_series = NumberSeries(config.TERRITORY_ID_PREFIX, config.SERIES_ID_START, config.SERIES_ID_END)
        for territory in territory_data:
            territory.region_id = number_series.get_id()
        
        if progress_callback:
            progress_callback(100, 100)
        
        args = (Image.fromarray(territory_image), territory_image, territory_data)
        if callable(getattr(self, "on_territories_generated", None)):
            self.on_territories_generated(*args)
        return args

    def _generate_provinces(self,
        territory_image: NDArray[np.uint8], territory_data: list[RegionMetadata],
        progress_callback=None,
    ) -> tuple[Image.Image, NDArray[np.uint8], list[RegionMetadata]]:
        def province_progress(current: int, total: int) -> None:
            if progress_callback:
                # Map progress (0-100) to overall progress (0-90)
                progress_callback(int((current / total) * 90), 100)
        
        province_image, province_data = convert_all_cont_areas_to_regions(
            cont_area_image=territory_image,
            cont_areas_metadata=territory_data,
            density_image=self.boundary_image,
            pixels_per_land_region=self.pixels_per_land_province,
            pixels_per_water_region=self.pixels_per_water_province,
            fn_new_number_series=lambda territory_meta: NumberSeries(
                f"{territory_meta.region_id}-TEMP", config.SERIES_ID_START, config.SERIES_ID_END
            ),
            rng_seed=self.provinces_rng_seed,
            lloyd_iterations=self.lloyd_iterations,
            override_density_multiplier=False,
            tqdm_description="Generating provinces from territories",
            tqdm_unit="territories",
            progress_callback=province_progress,
        )

        if progress_callback:
            progress_callback(100, 100)

        number_series = NumberSeries(config.PROVINCE_ID_PREFIX, config.SERIES_ID_START, config.SERIES_ID_END)
        for territory in territory_data:
            territory.region_id = number_series.get_id()

        args = (Image.fromarray(province_image), province_image, province_data)
        if callable(getattr(self, "on_provinces_generated", None)):
            self.on_provinces_generated(*args)
        return args


    def on_cont_areas_generated(self,
        cont_area_image: Image.Image, cont_area_image_buffer: NDArray[np.uint8], cont_area_data: list[RegionMetadata]) -> None: ...
    def on_type_classification_generated(self,
        class_image: Image.Image, class_image_buffer: NDArray[np.uint8], class_counts: dict[str, int]) -> None: ...
    def on_districts_generated(self,
        district_image: Image.Image, district_image_buffer: NDArray[np.uint8], district_data: list[RegionMetadata]) -> None: ...
    def on_territories_generated(self,
        territory_image: Image.Image, territory_image_buffer: NDArray[np.uint8], territory_data: list[RegionMetadata]) -> None: ...
    def on_provinces_generated(self,
        province_image: Image.Image, province_image_buffer: NDArray[np.uint8], province_data: list[RegionMetadata]) -> None: ...

    
    @staticmethod
    def clean_land_image(land_image: Image.Image) -> Image.Image:
        """
        Standardize a land image to configured ocean/lake/land colors.

        Args:
            land_image: Input land image as PIL Image.

        Returns:
            RGBA image where each pixel is reassigned to the nearest of
            `config.OCEAN_COLOR`, `config.LAKE_COLOR`, or `config.LAND_COLOR`.
        """
        class_image = classify_pixels_by_color(np.array(land_image.convert("RGBA")))
        return Image.fromarray(class_image)

    @staticmethod
    def clean_boundary_image(boundary_image: Image.Image) -> Image.Image:
        """
        Standardize a boundary image to strict black borders and gray areas.

        Args:
            boundary_image: Input boundary image as PIL Image.

        Returns:
            Image with boundaries set to (0, 0, 0, 255)
            and all other pixels set to (128, 128, 128, 255).
        """
        boundary_image = clean_boundary_image(np.array(boundary_image.convert("RGBA")))
        return Image.fromarray(boundary_image)
