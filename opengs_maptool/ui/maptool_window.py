from PIL import Image
from typing import Callable
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QPushButton, QMessageBox, QSpinBox, QSizePolicy
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QCloseEvent

from opengs_maptool.logic import MapTool
from opengs_maptool.ui.buttons import create_slider, create_button
from opengs_maptool.ui.image_display import ImageDisplay
from opengs_maptool.ui.flappy_bird_game import start_flappy_bird_process
from opengs_maptool import config


class ProgressButton(QPushButton):
    """Button with integrated progress bar background."""
    
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._progress = 0
        self._default_text = text
        self._is_processing = False
        self.setMinimumHeight(35)
        
    def set_progress(self, value: int) -> None:
        """Set progress value (0-100)."""
        self._progress = max(0, min(100, value))
        if not self._is_processing and value > 0:
            self._is_processing = True
            self.setText(f"{self._default_text} - {self._progress}%")
        elif self._is_processing:
            self.setText(f"{self._default_text} - {self._progress}%")
        self.update()  # Trigger repaint
        
    def reset_progress(self) -> None:
        """Reset progress to 0."""
        self._progress = 0
        self._is_processing = False
        self.setText(self._default_text)
        self.update()
        
    def paintEvent(self, event) -> None:
        """Custom paint to show progress as button background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background for the unfilled portion (light gray)
        if self._is_processing:
            background_color = QColor(220, 220, 220, 80)
            painter.fillRect(0, 0, self.width(), self.height(), background_color)
        
        # Draw progress background
        if self._progress > 0:
            progress_width = int((self.width() * self._progress) / 100)
            progress_color = QColor(70, 160, 70, 120)  # Green, semi-transparent
            painter.fillRect(0, 0, progress_width, self.height(), progress_color)
        
        # Let the default button paint on top
        painter.end()
        super().paintEvent(event)


class BackgroundWorker(QThread):
    """Generalized worker thread for running tasks in the background with progress tracking."""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    error = pyqtSignal(Exception)
    
    def __init__(self, task: Callable, *args, **kwargs) -> None:
        """
        Initialize the background worker.
        
        Args:
            task: The function to run in the background. 
                  If the task needs progress tracking, it should accept a 'progress_callback' kwarg.
            *args: Positional arguments to pass to the task
            **kwargs: Keyword arguments to pass to the task
        """
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = kwargs
    
    def run(self) -> None:
        """Run the task in a background thread."""
        try:
            # Inject progress callback if task accepts it
            def progress_callback(current: int, total: int = 100) -> None:
                percentage = int((current / total) * 100) if total > 0 else current
                self.progress.emit(percentage)
            
            # Add progress_callback to kwargs if not already present
            if 'progress_callback' not in self.kwargs:
                self.kwargs['progress_callback'] = progress_callback
            
            result = self.task(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(e)


EMPTY_IMAGE = Image.new("RGB", (16, 9), color=(100, 100, 100))

class MapToolWindow(QWidget):
    """
    Open Grand Strategy Map Tool, which can be used from a UI Window.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize data storage
        self._cont_area_image_buffer = None
        self._cont_area_data = None
        self._dens_samp_image_buffer = None
        self._dens_samp_data = None
        self._territory_image_buffer = None
        self._territory_data = None
        self._province_image_buffer = None
        self._province_data = None
        self.flappy_bird_process = None
        self.create_layout()
        self.showMaximized()
    

    def create_layout(self) -> None:
        # MAIN LAYOUT
        self.setWindowTitle(config.TITLE)
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        # Bottom bar with game button and version label
        bottom_layout = QHBoxLayout()
        self.button_flappy_bird = QPushButton("Play 🐦 While Waiting")
        self.button_flappy_bird.clicked.connect(self.on_button_play_flappy_bird)
        bottom_layout.addWidget(self.button_flappy_bird)
        bottom_layout.addStretch()
        self.label_version = QLabel("Version "+config.VERSION)
        bottom_layout.addWidget(self.label_version)
        main_layout.addLayout(bottom_layout)

        self.create_start_tab()
        self.tabs.addTab(self.readme_tab, "Getting Started")
        self.create_boundary_tab()
        self.tabs.addTab(self.boundary_tab, "Create Density Image")
        self.create_input_images_tab()
        self.tabs.addTab(self.input_tab, "Input Images")
        self.create_areas_tab()
        self.tabs.addTab(self.areas_tab, "Generate Areas")
        self.create_dens_samp_tab()
        self.tabs.addTab(self.dens_samp_tab, "Generate Density Samples")
        self.create_territory_tab()
        self.tabs.addTab(self.territory_tab, "Generate Territories")
        self.create_province_tab()
        self.tabs.addTab(self.province_tab, "Generate Provinces")

    def on_button_play_flappy_bird(self) -> None:
        """Open Flappy Bird game in a separate process."""
        if self.flappy_bird_process is not None:
            if self.flappy_bird_process.is_alive():
                QMessageBox.information(self, "Flappy Bird", "Flappy Bird is already running.")
                return
            self.flappy_bird_process = None

        try:
            self.flappy_bird_process = start_flappy_bird_process()
        except Exception as error:
            self.flappy_bird_process = None
            QMessageBox.critical(self, "Flappy Bird", f"Failed to start Flappy Bird: {error}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.flappy_bird_process is not None and self.flappy_bird_process.is_alive():
            self.flappy_bird_process.terminate()
            self.flappy_bird_process.join(timeout=1.0)
            self.flappy_bird_process = None
        super().closeEvent(event)

    # TAB 1
    def create_start_tab(self) -> None:
        self.readme_tab = QWidget()
        start_layout = QVBoxLayout(self.readme_tab)
        self.readme_label = QLabel(
            '<h1>Please read the README</h1>'
            '<h2><a href="https://github.com/Thomas-Holtvedt/opengs-maptool/blob/main/README.md">'        
            'Open the README in your browser</a></h2>'
        )
        self.readme_label.setOpenExternalLinks(True)
        self.readme_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        start_layout.addWidget(self.readme_label)

    # TAB 2
    def create_boundary_tab(self) -> None:
        self.boundary_tab = QWidget()
        boundary_tab_layout = QVBoxLayout(self.boundary_tab)
        
        create_button(boundary_tab_layout, f"Import and Clean {config.BOUNDARY_IMAGE_FILENAME}", self.on_button_import_boundary)

        self.adapt_boundary_image_display = ImageDisplay(name=config.BOUNDARY_IMAGE_FILENAME)
        self.adapt_boundary_image_display.set_image(EMPTY_IMAGE)
        boundary_tab_layout.addWidget(self.adapt_boundary_image_display, stretch=1)
        
        instruction_label = QLabel(
            '<h3>Instructions:</h3>'
            '<p>1. Save the above boundary image</p>'
            '<p>2. Edit the image in an image editor (e.g., Paint.NET, Photoshop, GIMP)</p>'
            '<p>3. Change the greyscale values for different territory & province density (1-255)</p>'
            '<p>4. Greyscale value <b>1</b> results in 4x fewer provinces and Greyscale value <b>255</b> results in 4x more provinces</p>'
            '<p><b>Important:</b> Greyscale value <b>0 (black)</b> is reserved for boundaries and will be removed</p>'
            '<p>5. Upload the edited image in the next tab</p>'
        )
        instruction_label.setWordWrap(True)
        boundary_tab_layout.addWidget(instruction_label)
    
    # TAB 3
    def create_input_images_tab(self) -> None:
        self.input_tab = QWidget()
        input_tab_layout = QVBoxLayout(self.input_tab)
        
        boundary_button_row = QHBoxLayout()
        input_tab_layout.addLayout(boundary_button_row)
        create_button(boundary_button_row, f"Import {config.FINAL_BOUNDARY_IMAGE_FILENAME}", self.on_button_import_final_boundary)
        create_button(boundary_button_row, "Keep Generated Image", self.on_button_keep_generated_boundary)

        self.final_boundary_image_display = ImageDisplay(name=config.FINAL_BOUNDARY_IMAGE_FILENAME)
        self.final_boundary_image_display.set_image(EMPTY_IMAGE)
        input_tab_layout.addWidget(self.final_boundary_image_display, stretch=1)

        create_button(input_tab_layout, f"Import and Clean {config.CLASS_IMAGE_FILENAME}", self.on_button_import_class)
        self.class_image_display = ImageDisplay(name=config.CLASS_IMAGE_FILENAME)
        self.class_image_display.set_image(EMPTY_IMAGE)
        input_tab_layout.addWidget(self.class_image_display, stretch=1)

    # TAB 4-7
    def create_areas_tab(self) -> None:
        self.areas_tab = QWidget()
        areas_tab_layout = QVBoxLayout(self.areas_tab)

        self.cont_areas_rng_seed_input = self._create_seed_input(
            areas_tab_layout,
            "Continuous Areas RNG Seed:",
            int(1e6),
        )
        
        self.button_generate_areas = ProgressButton("Generate Continuous Areas")
        self.button_generate_areas.clicked.connect(self.on_button_generate_areas)
        areas_tab_layout.addWidget(self.button_generate_areas)

        self.areas_image_display = ImageDisplay(name=config.CONTINUOUS_AREA_IMAGE_FILENAME, csv_export=True)
        self.areas_image_display.set_image(EMPTY_IMAGE)
        areas_tab_layout.addWidget(self.areas_image_display, stretch=1)

    def create_dens_samp_tab(self) -> None:
        self.dens_samp_tab = QWidget()
        dens_samp_tab_layout = QVBoxLayout(self.dens_samp_tab)

        self.dens_samps_rng_seed_input = self._create_seed_input(
            dens_samp_tab_layout,
            "Density Samples RNG Seed:",
            int(1_500_000),
        )

        self.pixels_per_land_dens_samp_slider = create_slider(dens_samp_tab_layout,
            "Pixels per land density sample:",
            config.PIXELS_PER_LAND_DENS_SAMP_MIN,
            config.PIXELS_PER_LAND_DENS_SAMP_MAX,
            config.PIXELS_PER_LAND_DENS_SAMP_DEFAULT,
            config.PIXELS_PER_LAND_DENS_SAMP_TICK,
            config.PIXELS_PER_LAND_DENS_SAMP_STEP,
        )

        self.pixels_per_water_dens_samp_slider = create_slider(dens_samp_tab_layout,
            "Pixels per water density sample:",
            config.PIXELS_PER_WATER_DENS_SAMP_MIN,
            config.PIXELS_PER_WATER_DENS_SAMP_MAX,
            config.PIXELS_PER_WATER_DENS_SAMP_DEFAULT,
            config.PIXELS_PER_WATER_DENS_SAMP_TICK,
            config.PIXELS_PER_WATER_DENS_SAMP_STEP,
        )

        self.button_gen_dens_samps = ProgressButton("Generate Density Samples")
        self.button_gen_dens_samps.clicked.connect(self.on_button_generate_dens_samps)
        dens_samp_tab_layout.addWidget(self.button_gen_dens_samps)

        self.dens_samp_image_display = ImageDisplay(name=config.DENS_SAMP_IMAGE_FILENAME, csv_export=True)
        self.dens_samp_image_display.set_image(EMPTY_IMAGE)
        dens_samp_tab_layout.addWidget(self.dens_samp_image_display, stretch=1)

    def create_territory_tab(self) -> None:
        self.territory_tab = QWidget()
        territory_tab_layout = QVBoxLayout(self.territory_tab)

        self.territories_rng_seed_input = self._create_seed_input(
            territory_tab_layout,
            "Territories RNG Seed:",
            int(2e6),
        )

        # Buttons
        self.pixels_per_land_territory_slider = create_slider(territory_tab_layout,
            "Pixels per land territory:",
            config.PIXELS_PER_LAND_TERRITORY_MIN,
            config.PIXELS_PER_LAND_TERRITORY_MAX,
            config.PIXELS_PER_LAND_TERRITORY_DEFAULT,
            config.PIXELS_PER_LAND_TERRITORY_TICK,
            config.PIXELS_PER_LAND_TERRITORY_STEP,
        )

        self.pixels_per_water_territory_slider = create_slider(territory_tab_layout,
            "Pixels per water territory:",
            config.PIXELS_PER_WATER_TERRITORY_MIN,
            config.PIXELS_PER_WATER_TERRITORY_MAX,
            config.PIXELS_PER_WATER_TERRITORY_DEFAULT,
            config.PIXELS_PER_WATER_TERRITORY_TICK,
            config.PIXELS_PER_WATER_TERRITORY_STEP,
        )

        self.button_gen_territories = ProgressButton("Generate Territories")
        self.button_gen_territories.clicked.connect(self.on_button_generate_territories)
        territory_tab_layout.addWidget(self.button_gen_territories)

        self.territory_image_display = ImageDisplay(name=config.TERRITORY_IMAGE_FILENAME, csv_export=True)
        self.territory_image_display.set_image(EMPTY_IMAGE)
        territory_tab_layout.addWidget(self.territory_image_display, stretch=1)

    def create_province_tab(self) -> None:
        self.province_tab = QWidget()
        province_tab_layout = QVBoxLayout(self.province_tab)

        self.provinces_rng_seed_input = self._create_seed_input(
            province_tab_layout,
            "Provinces RNG Seed:",
            int(3e6),
        )

        # Buttons
        self.pixels_per_land_province_slider = create_slider(province_tab_layout,
            "Pixels per land province:",
            config.PIXELS_PER_LAND_PROVINCE_MIN,
            config.PIXELS_PER_LAND_PROVINCE_MAX,
            config.PIXELS_PER_LAND_PROVINCE_DEFAULT,
            config.PIXELS_PER_LAND_PROVINCE_TICK,
            config.PIXELS_PER_LAND_PROVINCE_STEP,
        )

        self.pixels_per_water_province_slider = create_slider(province_tab_layout,
            "Pixels per water province:",
            config.PIXELS_PER_WATER_PROVINCE_MIN,
            config.PIXELS_PER_WATER_PROVINCE_MAX,
            config.PIXELS_PER_WATER_PROVINCE_DEFAULT,
            config.PIXELS_PER_WATER_PROVINCE_TICK,
            config.PIXELS_PER_WATER_PROVINCE_STEP,
        )

        self.button_gen_provinces = ProgressButton("Generate Provinces")
        self.button_gen_provinces.clicked.connect(self.on_button_generate_provinces)
        province_tab_layout.addWidget(self.button_gen_provinces)
    
        self.province_image_display = ImageDisplay(name=config.PROVINCE_IMAGE_FILENAME, csv_export=True)
        self.province_image_display.set_image(EMPTY_IMAGE)
        province_tab_layout.addWidget(self.province_image_display, stretch=1)


    # TAB 2
    def on_button_import_boundary(self) -> None:
        if not self.adapt_boundary_image_display.import_image():
            return

        image = self.adapt_boundary_image_display.get_image()
        if image is None:
            return

        try:
            cleaned_image = MapTool.clean_boundary_image(image)
            self.adapt_boundary_image_display.set_image(cleaned_image)

        except Exception as error:
            QMessageBox.critical(self, "Error", f"Error processing classification image: {error}")

    # TAB 3
    def on_button_import_final_boundary(self) -> None:
        self.final_boundary_image_display.import_image()

    def on_button_keep_generated_boundary(self) -> None:
        self.final_boundary_image_display.set_image(self.adapt_boundary_image_display.get_image() or EMPTY_IMAGE)

    def on_button_import_class(self) -> None:
        if not self.class_image_display.import_image():
            return

        image = self.class_image_display.get_image()
        if image is None:
            return
        
        try:
            cleaned_class_image = MapTool.clean_class_image(image)
            self.class_image_display.set_image(cleaned_class_image)
        except Exception as error:
            QMessageBox.critical(self, "Error", f"Error processing classification image: {error}")

    # TAB 4-7
    def on_button_generate_areas(self) -> None:
        def run_task(maptool: MapTool, progress_callback: Callable) -> tuple:
            cont_area_image, cont_area_image_buffer, cont_area_data = maptool._generate_cont_areas(progress_callback=progress_callback)
            return (cont_area_image, cont_area_image_buffer, cont_area_data)
        
        def on_progress(value: int) -> None:
            self.button_generate_areas.set_progress(value)
        
        def on_finished(result: tuple) -> None:
            self.button_generate_areas.reset_progress()
            self.button_generate_areas.setEnabled(True)
            cont_area_image, cont_area_image_buffer, cont_area_data = result
            self.areas_image_display.set_image(cont_area_image)
            self.areas_image_display.set_data(cont_area_data, "Continuous Area Data")
            # Store for later use in territory/province generation
            self._cont_area_image_buffer = cont_area_image_buffer
            self._cont_area_data = cont_area_data
        
        def on_error(error: Exception) -> None:
            self.button_generate_areas.reset_progress()
            self.button_generate_areas.setEnabled(True)
            QMessageBox.critical(self, "Error", f"Error generating areas: {error}")

        self.button_generate_areas.reset_progress()
        self.button_generate_areas.setEnabled(False)
        
        self.areas_worker = self._create_background_worker(run_task, on_progress, on_finished, on_error)

    def on_button_generate_dens_samps(self) -> None:
        if self._cont_area_image_buffer is None:
            QMessageBox.warning(self, "Warning", "Continuous areas must be generated first")
            return

        def run_task(maptool: MapTool, progress_callback: Callable) -> tuple:
            dens_samp_image, dens_samp_image_buffer, dens_samp_data = maptool._generate_dens_samps(
                self._cont_area_image_buffer,
                self._cont_area_data,
                progress_callback=progress_callback,
            )
            return (dens_samp_image, dens_samp_image_buffer, dens_samp_data)

        def on_progress(value: int) -> None:
            self.button_gen_dens_samps.set_progress(value)

        def on_finished(result: tuple) -> None:
            self.button_gen_dens_samps.reset_progress()
            self.button_gen_dens_samps.setEnabled(True)
            dens_samp_image, dens_samp_image_buffer, dens_samp_data = result
            self.dens_samp_image_display.set_image(dens_samp_image)
            self.dens_samp_image_display.set_data(dens_samp_data, "Density Samples Data")
            self._dens_samp_image_buffer = dens_samp_image_buffer
            self._dens_samp_data = dens_samp_data

        def on_error(error: Exception) -> None:
            self.button_gen_dens_samps.reset_progress()
            self.button_gen_dens_samps.setEnabled(True)
            QMessageBox.critical(self, "Error", f"Error generating density samples: {error}")

        self.button_gen_dens_samps.reset_progress()
        self.button_gen_dens_samps.setEnabled(False)

        self.dens_samps_worker = self._create_background_worker(run_task, on_progress, on_finished, on_error)
    
    def on_button_generate_territories(self) -> None:
        if self._dens_samp_image_buffer is None:
            QMessageBox.warning(self, "Warning", "Density Samples must be generated first")
            return
        
        def run_task(maptool: MapTool, progress_callback: Callable) -> tuple:
            territory_image, territory_image_buffer, territory_data = maptool._generate_territories(
                self._dens_samp_image_buffer,
                self._dens_samp_data,
                progress_callback=progress_callback,
            )
            return (territory_image, territory_image_buffer, territory_data)
        
        def on_progress(value: int) -> None:
            self.button_gen_territories.set_progress(value)
        
        def on_finished(result: tuple) -> None:
            self.button_gen_territories.reset_progress()
            self.button_gen_territories.setEnabled(True)
            territory_image, territory_image_buffer, territory_data = result
            self.territory_image_display.set_image(territory_image)
            self.territory_image_display.set_data(territory_data, "Territory Data")
            # Store for later use in province generation
            self._territory_image_buffer = territory_image_buffer
            self._territory_data = territory_data
        
        def on_error(error: Exception) -> None:
            self.button_gen_territories.reset_progress()
            self.button_gen_territories.setEnabled(True)
            QMessageBox.critical(self, "Error", f"Error generating territories: {error}")

        self.button_gen_territories.reset_progress()
        self.button_gen_territories.setEnabled(False)
        
        self.territories_worker = self._create_background_worker(run_task, on_progress, on_finished, on_error)

    def on_button_generate_provinces(self) -> None:
        if self._territory_image_buffer is None:
            QMessageBox.warning(self, "Warning", "Territories must be generated first")
            return
        
        def run_task(maptool: MapTool, progress_callback: Callable) -> tuple:
            province_image, province_image_buffer, province_data = maptool._generate_provinces(
                self._territory_image_buffer,
                self._territory_data,
                progress_callback=progress_callback,
            )
            return (province_image, province_image_buffer, province_data)
        
        def on_progress(value: int) -> None:
            self.button_gen_provinces.set_progress(value)
        
        def on_finished(result: tuple) -> None:
            self.button_gen_provinces.reset_progress()
            self.button_gen_provinces.setEnabled(True)
            province_image, province_image_buffer, province_data = result
            self.province_image_display.set_image(province_image)
            self.province_image_display.set_data(province_data, "Province Data")
            # Store for later use
            self._province_image = province_image
            self._province_image_buffer = province_image_buffer
            self._province_data = province_data
        
        def on_error(error: Exception) -> None:
            self.button_gen_provinces.reset_progress()
            self.button_gen_provinces.setEnabled(True)
            QMessageBox.critical(self, "Error", f"Error generating provinces: {error}")

        self.button_gen_provinces.reset_progress()
        self.button_gen_provinces.setEnabled(False)
        
        self.provinces_worker = self._create_background_worker(run_task, on_progress, on_finished, on_error)


    def _create_maptool(self) -> MapTool:
        return MapTool(
            class_image=self.class_image_display.get_image(),
            boundary_image=self.final_boundary_image_display.get_image(),
            pixels_per_land_territory=self.pixels_per_land_territory_slider.value(),
            pixels_per_water_territory=self.pixels_per_water_territory_slider.value(),
            pixels_per_land_province=self.pixels_per_land_province_slider.value(),
            pixels_per_water_province=self.pixels_per_water_province_slider.value(),
            pixels_per_land_dens_samp=self.pixels_per_land_dens_samp_slider.value(),
            pixels_per_water_dens_samp=self.pixels_per_water_dens_samp_slider.value(),
            cont_areas_rng_seed=self.cont_areas_rng_seed_input.value(),
            dens_samps_rng_seed=self.dens_samps_rng_seed_input.value(),
            territories_rng_seed=self.territories_rng_seed_input.value(),
            provinces_rng_seed=self.provinces_rng_seed_input.value(),
        )

    def _create_seed_input(self, parent_layout: QVBoxLayout, label_text: str, default_value: int) -> QSpinBox:
        row = QHBoxLayout()
        parent_layout.addLayout(row)

        label = QLabel(label_text)
        row.addWidget(label)

        spinbox = QSpinBox()
        spinbox.setMinimum(0)
        spinbox.setMaximum(2_147_483_647)
        spinbox.setSingleStep(1)
        spinbox.setValue(default_value)
        row.addWidget(spinbox)

        return spinbox
    
    def _create_background_worker(self, run_task: Callable, on_progress: Callable, on_finished: Callable, on_error: Callable) -> BackgroundWorker:
        worker = BackgroundWorker(run_task, self._create_maptool())
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.start()
        return worker

