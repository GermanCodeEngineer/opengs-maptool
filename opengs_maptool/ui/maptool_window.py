import logging
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel,
    QPushButton, QMessageBox, QSpinBox, QSizePolicy, QScrollArea,
)
import traceback
from typing import Callable

from opengs_maptool.logic import StepMapTool
from opengs_maptool.ui.buttons import create_slider, create_button, ProgressButton
from opengs_maptool.ui.image_display import ImageDisplay, EMPTY_IMAGE
from opengs_maptool.ui.flappy_bird_game import start_flappy_bird_process
from opengs_maptool import config


def log_error_with_traceback(error: Exception, message: str) -> None:
    try: # Use a trick to insert the message before the error
        raise RuntimeError(f"{message}: {error}") from error
    except RuntimeError as runtime_error:
        tb_str = "".join(traceback.format_exception(type(runtime_error), runtime_error, runtime_error.__traceback__))
        logging.error(f" {tb_str}")


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
                  If the task needs progress tracking, it should accept a "progress_callback" kwarg.
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
            if "progress_callback" not in self.kwargs:
                self.kwargs["progress_callback"] = progress_callback
            
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
        content_widget = QWidget()
        start_layout = QVBoxLayout(content_widget)
        self.readme_label = QLabel(
            "<h1>Please read the README</h1>"
            '<h2><a href="https://github.com/Thomas-Holtvedt/opengs-maptool/blob/main/README.md">'        
            "Open the README in your browser</a></h2>"
        )
        self.readme_label.setOpenExternalLinks(True)
        self.readme_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        start_layout.addWidget(self.readme_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        self.readme_tab = scroll

    # TAB 2
    def create_boundary_tab(self) -> None:
        
        content_widget = QWidget()
        boundary_tab_layout = QVBoxLayout(content_widget)
        create_button(boundary_tab_layout, f"Import and Clean {config.BOUNDARY_IMAGE_FILENAME}", self.on_button_import_boundary)

        self.adapt_boundary_image_display = ImageDisplay(name=config.BOUNDARY_IMAGE_FILENAME)
        self.adapt_boundary_image_display.setMinimumHeight(int(self.height() * 0.7) if self.height() > 0 else 200)
        boundary_tab_layout.addWidget(self.adapt_boundary_image_display, stretch=1)

        instruction_label = QLabel(
            "<h3>Instructions:</h3>"
            "<p>1. Save the above boundary image</p>"
            "<p>2. Edit the image in an image editor (e.g., Paint.NET, Photoshop, GIMP)</p>"
            "<p>3. Change the greyscale values for different territory and province density (1-255)</p>"
            "<p>4. Greyscale value <b>1</b> results in 4x fewer provinces and Greyscale value <b>255</b> results in 4x more provinces</p>"
            "<p><b>Important:</b> Greyscale value <b>0 (black)</b> is reserved for boundaries and will be removed</p>"
            "<p>5. Upload the edited image in the next tab</p>"
        )
        instruction_label.setWordWrap(True)
        boundary_tab_layout.addWidget(instruction_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        self.boundary_tab = scroll
    
    # TAB 3
    def create_input_images_tab(self) -> None:
        content_widget = QWidget()
        input_tab_layout = QVBoxLayout(content_widget)

        boundary_button_row = QHBoxLayout()
        input_tab_layout.addLayout(boundary_button_row)
        create_button(boundary_button_row, f"Import {config.FINAL_BOUNDARY_IMAGE_FILENAME}", self.on_button_import_final_boundary)
        create_button(boundary_button_row, "Keep Generated Image", self.on_button_keep_generated_boundary)

        self.final_boundary_image_display = ImageDisplay(name=config.FINAL_BOUNDARY_IMAGE_FILENAME)
        self.final_boundary_image_display.setMinimumHeight(int(self.height() * 0.7) if self.height() > 0 else 200)
        input_tab_layout.addWidget(self.final_boundary_image_display, stretch=1)

        create_button(input_tab_layout, f"Import and Clean {config.CLASS_IMAGE_FILENAME}", self.on_button_import_class)
        self.class_image_display = ImageDisplay(name=config.CLASS_IMAGE_FILENAME)
        self.class_image_display.setMinimumHeight(int(self.height() * 0.7) if self.height() > 0 else 200)
        input_tab_layout.addWidget(self.class_image_display, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        self.input_tab = scroll

    # TAB 4-7
    def create_areas_tab(self) -> None:
        content_widget = QWidget()
        areas_tab_layout = QVBoxLayout(content_widget)

        self.button_generate_areas = ProgressButton("Generate Continuous Areas")
        self.button_generate_areas.clicked.connect(self.on_button_generate_areas)
        areas_tab_layout.addWidget(self.button_generate_areas)

        self.area_image_display = ImageDisplay(name=config.CONTINUOUS_AREA_IMAGE_FILENAME, csv_export=True)
        self.area_image_display.setMinimumHeight(int(self.height() * 0.7) if self.height() > 0 else 200)
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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        self.areas_tab = scroll

    def create_territory_tab(self) -> None:
        content_widget = QWidget()
        territory_tab_layout = QVBoxLayout(content_widget)

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
        self.territory_image_display.setMinimumHeight(int(self.height() * 0.7) if self.height() > 0 else 200)
        territory_tab_layout.addWidget(self.territory_image_display, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        self.territory_tab = scroll

    def create_province_tab(self) -> None:
        content_widget = QWidget()
        province_tab_layout = QVBoxLayout(content_widget)

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
        self.province_image_display.setMinimumHeight(int(self.height() * 0.7) if self.height() > 0 else 200)
        province_tab_layout.addWidget(self.province_image_display, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        self.province_tab = scroll


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
        self._start_generation(
            button=self.button_generate_areas,
            worker_attr="areas_worker",
            run_func=StepMapTool.generate_cont_areas,
            display=self.area_image_display,
            display_setter="set_image_buffer",
            data_setter="set_data",
            data_label="Continuous Area Data",
            kwargs=dict(
                class_image=self.class_image_display.get_image_buffer(),
                boundary_image=self.final_boundary_image_display.get_image_buffer(),
                rng_seed=config.DEFAULT_CONT_AREAS_RNG_SEED,
            )
        )

    def on_button_generate_dens_samps(self) -> None:
        if self.area_image_display.get_data() is None:
            QMessageBox.warning(self, "Warning", "Continuous areas must be generated first")
            return
        self._start_generation(
            button=self.button_gen_dens_samps,
            worker_attr="_dens_samps_worker",
            run_func=StepMapTool.generate_dens_samps,
            display=None,
            display_setter=None,
            data_setter=None,
            data_label=None,
            extra_result_handler=lambda result: setattr(self, "_dens_samp_image_buffer", result[0]) or setattr(self, "_dens_samp_data", result[1]),
            kwargs=dict(
                boundary_image=self.final_boundary_image_display.get_image_buffer(),
                cont_area_image=self.area_image_display.get_image_buffer(),
                cont_area_data=self.area_image_display.get_data(),
                pixels_per_land_dens_samp=self.pixels_per_land_dens_samp_slider.value(),
                pixels_per_water_dens_samp=self.pixels_per_water_dens_samp_slider.value(),
                rng_seed=config.DEFAULT_DENS_SAMPS_RNG_SEED,
            )
        )
    
    def on_button_generate_territories(self) -> None:
        if self._dens_samp_data is None:
            QMessageBox.warning(self, "Warning", "Density Samples must be generated first")
            return
        self._start_generation(
            button=self.button_gen_territories,
            worker_attr="_dens_samps_worker",
            run_func=StepMapTool.generate_territories,
            display=self.territory_image_display,
            display_setter="set_image_buffer",
            data_setter="set_data",
            data_label="Territory Data",
            kwargs=dict(
                boundary_image=self.final_boundary_image_display.get_image_buffer(),
                dens_samp_image=self._dens_samp_image_buffer,
                dens_samp_data=self._dens_samp_data,
                pixels_per_land_territory=self.pixels_per_land_territory_slider.value(),
                pixels_per_water_territory=self.pixels_per_water_territory_slider.value(),
                rng_seed=config.DEFAULT_TERRITORIES_RNG_SEED,
            )
        )

    def on_button_generate_provinces(self) -> None:
        if self.territory_image_display.get_data() is None:
            QMessageBox.warning(self, "Warning", "Territories must be generated first")
            return
        self._start_generation(
            button=self.button_gen_provinces,
            worker_attr="_dens_samps_worker",
            run_func=StepMapTool.generate_provinces,
            display=self.province_image_display,
            display_setter="set_image_buffer",
            data_setter="set_data",
            data_label="Province Data",
            kwargs=dict(
                boundary_image=self.final_boundary_image_display.get_image_buffer(),
                cont_area_image=self.territory_image_display.get_image_buffer(),
                cont_area_data=self.territory_image_display.get_data(),
                pixels_per_land_province=self.pixels_per_land_province_slider.value(),
                pixels_per_water_province=self.pixels_per_water_province_slider.value(),
                rng_seed=config.DEFAULT_TERRITORIES_RNG_SEED,
            )
        )


    def _start_generation(
        self,
        button: QPushButton,
        worker_attr: str,
        run_func: Callable,
        display: ImageDisplay,
        display_setter: str,
        data_setter: str,
        data_label: str,
        kwargs,
        extra_result_handler: Callable | None = None,
    ) -> None:
        """
        Helper to start a background generation task with progress and error handling.

        Args:
            button: The ProgressButton to update and disable during processing.
            worker_attr: Attribute name to store the worker thread on self.
            run_func: The function to run in the background (should accept progress_callback).
            display: The ImageDisplay widget to update with results (can be None).
            display_setter: Name of the method to set the image buffer (e.g., "set_image_buffer").
            data_setter: Name of the method to set the data (e.g., "set_data").
            data_label: Label for the data (used in UI and error messages).
            kwargs: Arguments to pass to run_func.
            extra_result_handler: Optional function to handle the result tuple for custom logic.
        """

        # Wrap the run_func to inject the progress_callback argument
        def run_task(progress_callback: Callable, **kw):
            return run_func(**kw, progress_callback=progress_callback)

        # Update the button's progress bar
        def on_progress(value: int):
            button.set_progress(value)

        # Handle successful completion: update UI and call any extra handler
        def on_finished(result: tuple):
            # Only show 100% progress after all post-processing is done
            # Set image buffer if applicable
            if display and display_setter and hasattr(display, display_setter):
                getattr(display, display_setter)(result[0])
            # Set data if applicable
            if display and data_setter and hasattr(display, data_setter):
                getattr(display, data_setter)(result[1], data_label)
            # Call any extra handler (e.g., for storing results on self)
            if extra_result_handler:
                extra_result_handler(result)
            button.set_progress(100)
            button.reset_progress()
            button.setEnabled(True)

        # Handle errors: reset UI and show error dialog
        def on_error(error: Exception):
            button.reset_progress()
            button.setEnabled(True)
            log_error_with_traceback(error, f"Error generating {data_label or ''}".strip())
            QMessageBox.critical(self, "Error", f"Error generating {data_label or ''}: {error}".strip())

        # Prepare UI and start the worker
        button.reset_progress()
        button.setEnabled(False)
        worker = self._create_background_worker(run_task, on_progress, on_finished, on_error, **kwargs)
        setattr(self, worker_attr, worker)

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
