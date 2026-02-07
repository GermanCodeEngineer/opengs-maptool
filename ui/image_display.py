import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PyQt6.QtWidgets import QLabel, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import config


class ImageDisplay(QLabel):
    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #333")
        self.setScaledContents(False)

        self._image = None
        self._original_pixmap = None
        self.name = name

    def set_image(self, image: Image.Image) -> None:
        self._image = image
        qimage = QImage(
            image.tobytes("raw", "RGBA"),
            image.width,
            image.height,
            QImage.Format.Format_RGBA8888
        )
        self._original_pixmap = QPixmap.fromImage(qimage)
        self._scale_image_to_fit()

    def _scale_image_to_fit(self) -> None:
        if self._original_pixmap is None:
            return
        
        pixmap = self._original_pixmap.scaled(
            self.width(),
            self.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scale_image_to_fit()

    def get_image(self) -> Image.Image | None:
        return self._image
    
    def get_image_buffer(self) -> NDArray[np.uint8] | None:
        if self._image is None:
            return None
        return np.array(self._image.convert("RGBA"), dtype=np.uint8)
    

    def import_image(self) -> None:
        Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Import {self.name}",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if not path:
            return

        imported_image = Image.open(path)
        self.set_image(imported_image)
