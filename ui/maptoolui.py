from PIL import Image
from logic.maptool import MapTool


class MapToolUI(MapTool):
    """
    Open Grand Strategy Map Tool, which can be used from a UI Window.
    """
    def __init__(self, land_image: Image.Image, boundary_image: Image.Image) -> None:
        """
        Args:
            land_image: Land/Ocean/Lake image
            boundary_image: Boundary and Region Density image
        """
        super().__init__(land_image, boundary_image)
    
    # WORK IN PROGRESS
