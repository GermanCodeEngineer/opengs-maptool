from io import BytesIO
import json
import numpy as np
from opengs_maptool.models.project import Project
from PIL import Image
import zipfile

class ProjectService:
    """Create, load, and save Project instances using the GSMAP/ZIP format."""

    def create(self) -> Project:
        """
        Create a new empty project.
        """
        project = Project()
        return project


    def load(self, path: str) -> Project:
        """
        Load a project saved to disk (GSMAP file or ZIP)

        @param path: The file path
        """
        with zipfile.ZipFile(path, "r") as zip:
            details = json.loads(zip.read("project.json"))

            # Load project's main details
            project = Project(
                name=details['name'],
                editor_version=details['editor_version'],
                description=details.get('description'),
                author=details.get('author')
            )
            project.file_path = path

            # GCE-TODO: implement deserialization
            # Load map images
            project.land_image = self._load_image_from_zip(zip, "land.png")
            project.boundary_image = self._load_image_from_zip(zip, "boundary.png")
            project.density_image = self._load_image_from_zip(zip, "density.png")
            project.terrain_image = self._load_image_from_zip(zip, "terrain.png")
            project.territory_image = self._load_image_from_zip(zip, "territory.png")
            project.province_image = self._load_image_from_zip(zip, "province.png")

            # Load data
            project.territory_data = self._load_data_from_zip(zip, "territory_data.json")
            project.province_data = self._load_data_from_zip(zip, "province_data.json")

            # Load metadata
            project.territory_pmap = self._load_territory_pmap_from_zip(zip)
            project.cached_masks = self._load_cached_masks_from_zip(zip)

            return project


    def save(self, project: Project, path: str):
        """
        Save a project to disk (GSMAP file or ZIP)

        @param path: The file path
        """
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zip:
            project.file_path = path

            # Save the project's main details
            details = {
                "name": project.name,
                "editor_version": project.editor_version,
                "description": project.description,
                "author": project.author
            }

            zip.writestr("project.json", json.dumps(details, indent=4))

            # TODO: implement serialization
            # Save map images
            self._save_image_in_zip(zip, project.land_image, "land.png")
            self._save_image_in_zip(zip, project.boundary_image, "boundary.png")
            self._save_image_in_zip(zip, project.density_image, "density.png")
            self._save_image_in_zip(zip, project.terrain_image, "terrain.png")
            self._save_image_in_zip(zip, project.territory_image, "territory.png")
            self._save_image_in_zip(zip, project.province_image, "province.png")

            # Save map data
            self._save_data_in_zip(zip, project.territory_data, "territory_data.json")
            self._save_data_in_zip(zip, project.province_data, "province_data.json")

            # Save metadata
            if project.territory_pmap is not None:
                buffer = BytesIO()

                np.save(buffer, project.territory_pmap)
                zip.writestr("metadata/territory_pmap.npy", buffer.getvalue())

            if project.cached_masks is not None:
                buffer = BytesIO()

                np.savez(buffer, **project.cached_masks)
                zip.writestr("metadata/cached_masks.npz", buffer.getvalue())

            # Update dirty indicator
            project.modified = False


    def _load_image_from_zip(self, zip, filename) -> Image:
        """
        Private method for loading an image contained in a GSMAP or ZIP file

        @param zip: The GSMAP or ZIP file
        @param filename: The image name (example: land.png)
        """
        try:
            image_data = zip.read("images/" + filename)

        except KeyError:
            return None

        return Image.open(
            BytesIO(image_data)
        ).copy()


    def _load_data_from_zip(self, zip, filename) -> dict:
        """
        Private method for loading a JSON file contained in a GSMAP or ZIP file

        @param zip: The GSMAP or ZIP file
        @param filename: The JSON file name (example: territory_data.json)
        """
        try:
            json_data = zip.read("data/" + filename)

        except KeyError:
            return None

        return json.load(
            BytesIO(json_data)
        )


    def _load_territory_pmap_from_zip(self, zip) -> np.ndarray:
        """
        Private method to load 'territory_pmap.npy' file contained in a GSMAP or ZIP file

        @param zip: The GSMAP or ZIP file
        """
        try:
            territory_pmap_data = zip.read("metadata/territory_pmap.npy")

        except KeyError:
            return None

        return np.load(BytesIO(territory_pmap_data))


    def _load_cached_masks_from_zip(self, zip) -> dict:
        """
        Private method to load 'cached_masks.npz' file contained in a GSMAP or ZIP file

        @param zip: The GSMAP or ZIP file
        """
        try:
            cached_masks_data = zip.read("metadata/cached_masks.npz")
            extracted_cached_masks = np.load(BytesIO(cached_masks_data))

        except KeyError:
            return None

        return {key: extracted_cached_masks[key] for key in extracted_cached_masks.files}


    def _save_image_in_zip(self, zip, image, filename):
        """
        Private method for saving an image to a GSMAP or ZIP file.

        @param zip: The GSMAP or ZIP file
        @param image: The image
        @param filename: The image name (example: density.png)
        """
        if image == None:
            return

        buffer = BytesIO()

        image.save(buffer, format="PNG")
        zip.writestr("images/" + filename, buffer.getvalue())


    def _save_data_in_zip(self, zip, data, filename):
        """
        Private method for saving data (dictionary) in JSON format to a GSMAP or ZIP file.

        @param zip: The GSMAP or ZIP file
        @param image: The data
        @param filename: The file name (example: province_data.json)
        """
        if data == None:
            return

        zip.writestr("data/" + filename, json.dumps(data, indent=4))
