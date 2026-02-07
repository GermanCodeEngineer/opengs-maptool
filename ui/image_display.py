from PIL import Image
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import config


class ImageDisplay(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(config.DISPLAY_SIZE_WIDTH,
                            config.DISPLAY_SIZE_HEIGHT)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #333")

        self._image = None

    def set_image(self, image: Image.Image) -> None:
        self._image = image
        qimage = QImage(
            image.tobytes("raw", "RGBA"),
            image.width,
            image.height,
            QImage.Format.Format_RGBA8888
        )
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio)

        self.setPixmap(pixmap)

    def get_image(self) -> Image.Image:
        return self._image
