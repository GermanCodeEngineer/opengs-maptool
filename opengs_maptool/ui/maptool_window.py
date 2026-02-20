import logging
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QPushButton, QMessageBox, QSpinBox, QSizePolicy
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QCloseEvent
import traceback
from typing import Callable

from opengs_maptool.logic import StepMapTool
from opengs_maptool.ui.buttons import create_slider, create_button
from opengs_maptool.ui.image_display import ImageDisplay, EMPTY_IMAGE
from opengs_maptool.ui.flappy_bird_game import start_flappy_bird_process
from opengs_maptool import config


def log_error_with_traceback(error: Exception, message: str) -> None:
    try: # Use a trick to insert the message before the error
        raise RuntimeError(f"{message}: {error}") from error
    except RuntimeError as runtime_error:
        tb_str = ''.join(traceback.format_exception(type(runtime_error), runtime_error, runtime_error.__traceback__))
        logging.error(f" {tb_str}")

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


class MapToolWindow(QWidget):
    """
    Open Grand Strategy Map Tool, which can be used from a UI Window.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize data storage
        self._dens_samp_image_buffer = None
        self._dens_samp_data = None
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
        except Exception as error:
            log_error_with_traceback(error, "Failed to start Flappy Bird")
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
        boundary_tab_layout.addWidget(self.adapt_boundary_image_display, stretch=1)
        
        instruction_label = QLabel(
            '<h3>Instructions:</h3>'
            '<p>1. Save the above boundary image</p>'
            '<p>2. Edit the image in an image editor (e.g., Paint.NET, Photoshop, GIMP)</p>'
            '<p>3. Change the greyscale values for different territory and province density (1-255)</p>'
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
        input_tab_layout.addWidget(self.final_boundary_image_display, stretch=1)

        create_button(input_tab_layout, f"Import and Clean {config.CLASS_IMAGE_FILENAME}", self.on_button_import_class)
        self.class_image_display = ImageDisplay(name=config.CLASS_IMAGE_FILENAME)
        input_tab_layout.addWidget(self.class_image_display, stretch=1)

    # TAB 4-7
    def create_areas_tab(self) -> None:
        self.areas_tab = QWidget()
        areas_tab_layout = QVBoxLayout(self.areas_tab)
        
        self.button_generate_areas = ProgressButton("Generate Continuous Areas")
        self.button_generate_areas.clicked.connect(self.on_button_generate_areas)
        areas_tab_layout.addWidget(self.button_generate_areas)

        self.area_image_display = ImageDisplay(name=config.CONTINUOUS_AREA_IMAGE_FILENAME, csv_export=True)
        areas_tab_layout.addWidget(self.area_image_display, stretch=1)

        self.pixels_per_land_dens_samp_slider = create_slider(areas_tab_layout,
            "Pixels per land density sample:",
            config.PIXELS_PER_LAND_DENS_SAMP_MIN,
            config.PIXELS_PER_LAND_DENS_SAMP_MAX,
            config.PIXELS_PER_LAND_DENS_SAMP_DEFAULT,
            config.PIXELS_PER_LAND_DENS_SAMP_TICK,
            config.PIXELS_PER_LAND_DENS_SAMP_STEP,
        )

        self.pixels_per_water_dens_samp_slider = create_slider(areas_tab_layout,
            "Pixels per water density sample:",
            config.PIXELS_PER_WATER_DENS_SAMP_MIN,
            config.PIXELS_PER_WATER_DENS_SAMP_MAX,
            config.PIXELS_PER_WATER_DENS_SAMP_DEFAULT,
            config.PIXELS_PER_WATER_DENS_SAMP_TICK,
            config.PIXELS_PER_WATER_DENS_SAMP_STEP,
        )

        self.button_gen_dens_samps = ProgressButton("Generate Density Samples")
        self.button_gen_dens_samps.clicked.connect(self.on_button_generate_dens_samps)
        areas_tab_layout.addWidget(self.button_gen_dens_samps)

    def create_territory_tab(self) -> None:
        self.territory_tab = QWidget()
        territory_tab_layout = QVBoxLayout(self.territory_tab)

        self.territories_rng_seed_input = self._create_seed_input(
            territory_tab_layout,
            "Territories RNG Seed:",
            config.DEFAULT_TERRITORIES_RNG_SEED,
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
        territory_tab_layout.addWidget(self.territory_image_display, stretch=1)

    def create_province_tab(self) -> None:
        self.province_tab = QWidget()
        province_tab_layout = QVBoxLayout(self.province_tab)

        self.provinces_rng_seed_input = self._create_seed_input(
            province_tab_layout,
            "Provinces RNG Seed:",
            config.DEFAULT_PROVINCES_RNG_SEED,
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
        province_tab_layout.addWidget(self.province_image_display, stretch=1)


    # TAB 2
    def on_button_import_boundary(self) -> None:
        if not self.adapt_boundary_image_display.import_image():
            return

        image = self.adapt_boundary_image_display.get_image()
        if image is None:
            return

        try:
            cleaned_image = StepMapTool.clean_boundary_image(image)
            self.adapt_boundary_image_display.set_image(cleaned_image)

        except Exception as error:
            log_error_with_traceback(error, "Error processing boundary image")
            QMessageBox.critical(self, "Error", f"Error processing boundary image: {error}")

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
            cleaned_class_image = StepMapTool.clean_class_image(image)
            self.class_image_display.set_image(cleaned_class_image)
        except Exception as error:
            log_error_with_traceback(error, "Error processing classification image")
            QMessageBox.critical(self, "Error", f"Error processing classification image: {error}")

    # TAB 4-7
    def on_button_generate_areas(self) -> None:
        def run_task(progress_callback: Callable, **kwargs) -> tuple:
            return StepMapTool.generate_cont_areas(**kwargs, progress_callback=progress_callback)
        
        def on_progress(value: int) -> None:
            self.button_generate_areas.set_progress(value)
        
        def on_finished(result: tuple) -> None:
            self.button_generate_areas.reset_progress()
            self.button_generate_areas.setEnabled(True)
            cont_area_image_buffer, cont_area_data = result
            self.area_image_display.set_image_buffer(cont_area_image_buffer)
            self.area_image_display.set_data(cont_area_data, "Continuous Area Data")
        
        def on_error(error: Exception) -> None:
            self.button_generate_areas.reset_progress()
            self.button_generate_areas.setEnabled(True)
            log_error_with_traceback(error, "Error generating areas")
            QMessageBox.critical(self, "Error", f"Error generating areas: {error}")

        self.button_generate_areas.reset_progress()
        self.button_generate_areas.setEnabled(False)
        
        self.areas_worker = self._create_background_worker(
            run_task, on_progress, on_finished, on_error,
            class_image=self.class_image_display.get_image_buffer(),
            boundary_image=self.final_boundary_image_display.get_image_buffer(),
            rng_seed=config.DEFAULT_CONT_AREAS_RNG_SEED,
        )

    def on_button_generate_dens_samps(self) -> None:
        if self.area_image_display.get_data() is None:
            QMessageBox.warning(self, "Warning", "Continuous areas must be generated first")
            return

        def run_task(progress_callback: Callable, **kwargs) -> tuple:
            return StepMapTool.generate_dens_samps(**kwargs, progress_callback=progress_callback)

        def on_progress(value: int) -> None:
            self.button_gen_dens_samps.set_progress(value)

        def on_finished(result: tuple) -> None:
            self.button_gen_dens_samps.reset_progress()
            self.button_gen_dens_samps.setEnabled(True)
            self._dens_samp_image_buffer, self._dens_samp_data = result

        def on_error(error: Exception) -> None:
            self.button_gen_dens_samps.reset_progress()
            self.button_gen_dens_samps.setEnabled(True)
            log_error_with_traceback(error, "Error generating density samples")
            QMessageBox.critical(self, "Error", f"Error generating density samples: {error}")

        self.button_gen_dens_samps.reset_progress()
        self.button_gen_dens_samps.setEnabled(False)

        self.dens_samps_worker = self._create_background_worker(
            run_task, on_progress, on_finished, on_error,
            boundary_image=self.final_boundary_image_display.get_image_buffer(),
            cont_area_image=self.area_image_display.get_image_buffer(),
            cont_area_data=self.area_image_display.get_data(),
            pixels_per_land_dens_samp=self.pixels_per_land_dens_samp_slider.value(),
            pixels_per_water_dens_samp=self.pixels_per_water_dens_samp_slider.value(),
            rng_seed=config.DEFAULT_DENS_SAMPS_RNG_SEED,
        )
    
    def on_button_generate_territories(self) -> None:
        if self._dens_samp_data is None:
            QMessageBox.warning(self, "Warning", "Density Samples must be generated first")
            return
        
        def run_task(progress_callback: Callable, **kwargs) -> tuple:
            return StepMapTool.generate_territories(**kwargs, progress_callback=progress_callback)
        
        def on_progress(value: int) -> None:
            self.button_gen_territories.set_progress(value)
        
        def on_finished(result: tuple) -> None:
            self.button_gen_territories.reset_progress()
            self.button_gen_territories.setEnabled(True)
            territory_image_buffer, territory_data = result
            self.territory_image_display.set_image_buffer(territory_image_buffer)
            self.territory_image_display.set_data(territory_data, "Territory Data")
        
        def on_error(error: Exception) -> None:
            self.button_gen_territories.reset_progress()
            self.button_gen_territories.setEnabled(True)
            log_error_with_traceback(error, "Error generating territories")
            QMessageBox.critical(self, "Error", f"Error generating territories: {error}")

        self.button_gen_territories.reset_progress()
        self.button_gen_territories.setEnabled(False)
        
        self.dens_samps_worker = self._create_background_worker(
            run_task, on_progress, on_finished, on_error,
            boundary_image=self.final_boundary_image_display.get_image_buffer(),
            dens_samp_image=self._dens_samp_image_buffer,
            dens_samp_data=self._dens_samp_data,
            pixels_per_land_territory=self.pixels_per_land_territory_slider.value(),
            pixels_per_water_territory=self.pixels_per_water_territory_slider.value(),
            rng_seed=config.DEFAULT_TERRITORIES_RNG_SEED,
        )

    def on_button_generate_provinces(self) -> None:
        if self.territory_image_display.get_data() is None:
            QMessageBox.warning(self, "Warning", "Territories must be generated first")
            return
        
        def run_task(progress_callback: Callable, **kwargs) -> tuple:
            return StepMapTool.generate_provinces(**kwargs, progress_callback=progress_callback)
        
        def on_progress(value: int) -> None:
            self.button_gen_provinces.set_progress(value)
        
        def on_finished(result: tuple) -> None:
            self.button_gen_provinces.reset_progress()
            self.button_gen_provinces.setEnabled(True)
            province_image_buffer, province_data = result
            self.province_image_display.set_image_buffer(province_image_buffer)
            self.province_image_display.set_data(province_data, "Province Data")
        
        def on_error(error: Exception) -> None:
            self.button_gen_provinces.reset_progress()
            self.button_gen_provinces.setEnabled(True)
            log_error_with_traceback(error, "Error generating provinces")
            QMessageBox.critical(self, "Error", f"Error generating provinces: {error}")

        self.button_gen_provinces.reset_progress()
        self.button_gen_provinces.setEnabled(False)
        
        self.dens_samps_worker = self._create_background_worker(
            run_task, on_progress, on_finished, on_error,
            boundary_image=self.final_boundary_image_display.get_image_buffer(),
            cont_area_image=self.territory_image_display.get_image_buffer(),
            cont_area_data=self.territory_image_display.get_data(),
            pixels_per_land_province=self.pixels_per_land_province_slider.value(),
            pixels_per_water_province=self.pixels_per_water_province_slider.value(),
            rng_seed=config.DEFAULT_TERRITORIES_RNG_SEED,
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
    
    def _create_background_worker(self, run_task: Callable, on_progress: Callable, on_finished: Callable, on_error: Callable, *args, **kwargs) -> BackgroundWorker:
        worker = BackgroundWorker(run_task, *args, **kwargs)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.start()
        return worker
