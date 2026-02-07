import numpy as np
from numpy.typing import NDArray
from PIL import Image
import json
from PyQt6.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QLabel, QFileDialog, QPushButton
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt
import config


class ImageDisplay(QWidget):
    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        self.name = name
        self.setMinimumSize(400, 300)
        
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Image label - spans entire grid
        self._image_label = QLabel()
        self._image_label.setMinimumSize(400, 300)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #333")
        self._image_label.setScaledContents(False)
        layout.addWidget(self._image_label, 0, 0)
        
        # Button container for top-right buttons
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(5, 5, 5, 0)
        button_layout.setSpacing(5)
        button_layout.addStretch()
        
        # JSON export button
        self._download_json_button = QPushButton("{}")
        self._download_json_button.setMaximumSize(40, 40)
        self._download_json_button.setStyleSheet(
            "QPushButton { "
            "  background-color: rgba(0, 0, 0, 150); "
            "  border: none; "
            "  border-radius: 5px; "
            "  padding: 5px; "
            "} "
            "QPushButton:hover { background-color: rgba(0, 0, 0, 200); }"
        )
        self._download_json_button.clicked.connect(self._on_download_json)
        self._download_json_button.setVisible(False)  # Only show if data is set
        button_layout.addWidget(self._download_json_button)
        
        # Download button - overlaid in top right corner
        self._download_button = QPushButton()
        self._download_button.setIcon(QIcon.fromTheme("document-save"))
        self._download_button.setMaximumSize(40, 40)
        self._download_button.setStyleSheet(
            "QPushButton { "
            "  background-color: rgba(0, 0, 0, 150); "
            "  border: none; "
            "  border-radius: 5px; "
            "  padding: 5px; "
            "} "
            "QPushButton:hover { background-color: rgba(0, 0, 0, 200); }"
        )
        self._download_button.clicked.connect(self._on_download)
        button_layout.addWidget(self._download_button)
        
        layout.addWidget(button_container, 0, 0, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        self._image = None
        self._original_pixmap = None
        self._data = None
        self._data_name = "Data"

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
    
    def set_data(self, data: dict | list, data_name: str = "Data") -> None:
        """Set JSON-like data to be exported."""
        self._data = data
        self._data_name = data_name
        self._download_json_button.setVisible(True)

    def _scale_image_to_fit(self) -> None:
        if self._original_pixmap is None:
            return
        
        pixmap = self._original_pixmap.scaled(
            self._image_label.width(),
            self._image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self._image_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scale_image_to_fit()

    def get_image(self) -> Image.Image | None:
        return self._image
    
    def get_image_buffer(self) -> NDArray[np.uint8] | None:
        if self._image is None:
            return None
        return np.array(self._image.convert("RGBA"), dtype=np.uint8)
    
    def _on_download(self) -> None:
        """Save the current image to a file."""
        if self._image is None:
            return
        
        Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {self.name}",
            f"{self.name}.png",
            "PNG Images (*.png);;JPEG Images (*.jpg);;BMP Images (*.bmp)"
        )
        if path:
            self._image.save(path)
    
    def _on_download_json(self) -> None:
        """Save the current data as JSON."""
        if self._data is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {self._data_name}",
            f"{self._data_name}.json",
            "JSON Files (*.json)"
        )
        if path:
            with open(path, 'w') as f:
                json.dump(self._data, f, indent=2)

    def import_image(self) -> bool:
        Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Import {self.name}",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if not path:
            return False

        imported_image = Image.open(path)
        self.set_image(imported_image)
        return True
