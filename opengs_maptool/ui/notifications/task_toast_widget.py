import traceback
from PyQt6.QtCore import QTimer, QPropertyAnimation, QEasingCurve, QEvent
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QVBoxLayout, QGraphicsOpacityEffect, QWidget
)
from opengs_maptool.models.progress_status import ProgressStatus
from opengs_maptool.ui.modals.error_modal import ErrorModal


class TaskToastWidget(QFrame):
    def __init__(self, title: str, main_window: QWidget | None = None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("taskToast")
        self._title = title
        self._main_window = main_window
        self._last_error: BaseException | None = None
        self._theme = {}
        self._is_dark_theme: bool | None = None
        self._applying_theme = False

        self._apply_theme()

        # Set up opacity effect for smooth fade-outs
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)

        # Layout setup
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("titleLabel")

        self.status_label = QLabel("Starting...")
        self.status_label.setObjectName("statusLabel")

        self.close_btn = QPushButton("x")
        self.close_btn.setObjectName("closeBtn")
        self.close_btn.setFixedSize(22, 22)
        self.close_btn.clicked.connect(self._on_close_clicked)
        self.close_btn.hide()

        top_row.addWidget(self.title_label)
        top_row.addStretch()
        top_row.addWidget(self.close_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)

        # Bottom row for error action buttons
        self.bottom_row = QHBoxLayout()
        self.bottom_row.setSpacing(8)
        self.details_btn = QPushButton("Show Details")
        self.details_btn.setObjectName("detailsBtn")
        self.details_btn.clicked.connect(self.show_error_modal)
        self.details_btn.hide()

        self.bottom_row.addWidget(self.status_label)
        self.bottom_row.addStretch()
        self.bottom_row.addWidget(self.details_btn)

        layout.addLayout(top_row)
        layout.addLayout(self.bottom_row)
        layout.addWidget(self.progress_bar)

        # Internal state
        self._task_signals = None
        self._is_active = False
        self._cancel_requested = False
        self._current_phase_description = "Working"
        self._set_status_tone("muted", bold=False)

    def _is_dark_palette(self) -> bool:
        return self.palette().window().color().lightness() < 128

    @staticmethod
    def _blend(base: QColor, other: QColor, ratio: float) -> QColor:
        """Mix `other` into `base`. ratio 0.0 -> base, 1.0 -> other."""
        return QColor(
            round(base.red() * (1 - ratio) + other.red() * ratio),
            round(base.green() * (1 - ratio) + other.green() * ratio),
            round(base.blue() * (1 - ratio) + other.blue() * ratio),
        )

    def _build_theme(self, is_dark: bool) -> dict[str, str]:
        """Derive the toast colours from the active palette.

        The rest of the app is unstyled Fusion, so anything hardcoded here would
        read as a foreign element. Only the status tones are fixed hues, because
        they carry meaning that a neutral palette cannot express.
        """
        palette = self.palette()
        window = palette.color(QPalette.ColorRole.Window)
        text = palette.color(QPalette.ColorRole.WindowText)

        if is_dark:
            # Lift the card off the panel it sits on.
            card = window.lighter(118)
            hover = window.lighter(145)
            tones = {
                "success": "#66bb6a",
                "warning": "#ffb74d",
                "error": "#ef5350",
            }
        else:
            card = palette.color(QPalette.ColorRole.Base)
            hover = window.darker(108)
            tones = {
                "success": "#2e7d32",
                "warning": "#b26a00",
                "error": "#c62828",
            }

        muted = self._blend(text, card, 0.45)
        theme = {
            "card": card.name(),
            "border": palette.color(QPalette.ColorRole.Mid).name(),
            "text": text.name(),
            "muted": muted.name(),
            "disabled": self._blend(text, card, 0.65).name(),
            "hover": hover.name(),
            # A cancelled task is a neutral outcome, not a coloured one.
            "cancelled": muted.name(),
        }
        theme.update(tones)
        return theme

    def _apply_theme(self, force: bool = False) -> None:
        if self._applying_theme:
            return

        is_dark = self._is_dark_palette()
        if not force and self._is_dark_theme == is_dark:
            return

        self._applying_theme = True
        self._is_dark_theme = is_dark
        self._theme = self._build_theme(is_dark)
        t = self._theme
        try:
            # The progress bar and the details button are deliberately left
            # unstyled so they render exactly like every other Fusion widget.
            self.setStyleSheet(
                f"""
                QFrame#taskToast {{
                    background: {t['card']};
                    border: 1px solid {t['border']};
                    border-radius: 4px;
                }}
                QLabel {{
                    color: {t['text']};
                }}
                QLabel#titleLabel {{
                    font-weight: 600;
                }}
                QLabel#statusLabel {{
                    color: {t['muted']};
                }}
                QPushButton#closeBtn {{
                    border: none;
                    border-radius: 3px;
                    background: transparent;
                    color: {t['muted']};
                    font-weight: 700;
                }}
                QPushButton#closeBtn:hover {{
                    background: {t['hover']};
                    color: {t['text']};
                }}
                QPushButton#closeBtn:disabled {{
                    color: {t['disabled']};
                }}
                """
            )
        finally:
            self._applying_theme = False

    def _set_status_tone(self, tone: str, *, bold: bool = True) -> None:
        color = self._theme.get(tone, self._theme["muted"])
        weight = "700" if bold else "500"
        self.status_label.setStyleSheet(f"color: {color}; font-weight: {weight};")

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_theme()

    def show_error_modal(self) -> None:
        """Opens the modal dialog detailing the exception that occurred."""
        if self._last_error is None:
            return

        error_text = (
            f"Task '{self._title}' failed with error:\n"
            f"{str(self._last_error)}\n\n"
            + "".join(traceback.format_exception(self._last_error))
        )
        error_modal = ErrorModal(
            main_window=self._main_window,
            error_text=error_text
        )
        error_modal.exec()

    def on_phase_started(self, phase_description: str, status: ProgressStatus) -> None:
        self._current_phase_description = phase_description or "Working"

        # Keep progress visuals in sync when a phase starts before the next update tick.
        self.progress_bar.setRange(0, status.total_steps)
        self.progress_bar.setValue(status.completed_steps)

        if status.total_steps > 0:
            pct = int((status.completed_steps / status.total_steps) * 100)
            self.status_label.setText(f"{self._current_phase_description}... {pct}%")
        else:
            self.status_label.setText(f"{self._current_phase_description}...")
        self._set_status_tone("muted", bold=False)

    def on_progress_started(self, status: ProgressStatus) -> None:
        self.progress_bar.setRange(0, status.total_steps)
        self.progress_bar.setValue(status.completed_steps)
        self.status_label.setText(f"{self._current_phase_description}...")
        self._set_status_tone("muted", bold=False)
        self._is_active = True
        self.close_btn.show()

    def on_progress_updated(self, status: ProgressStatus, steps_increased: int) -> None:
        self.progress_bar.setRange(0, status.total_steps)
        self.progress_bar.setValue(status.completed_steps)

        if status.total_steps > 0:
            pct = int((status.completed_steps / status.total_steps) * 100)
            self.status_label.setText(f"{self._current_phase_description}... {pct}%")
            self._set_status_tone("muted", bold=False)

    def fade_out_and_close(self, duration_ms: int = 400) -> None:
        """Smoothly fades out the toast widget before removing it."""
        self.anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self.anim.setDuration(duration_ms)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.anim.finished.connect(self.close)
        self.anim.start()

    def set_task_signals(self, signals) -> None:
        """Give the toast access to the task's signals so it can request cancellation."""
        self._task_signals = signals

    def on_success(self, result: object) -> None:
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText("Completed")
        self._set_status_tone("success")
        self._is_active = False
        self.close_btn.show()

        QTimer.singleShot(2500, self.fade_out_and_close)

    def on_error(self, error: BaseException) -> None:
        self._last_error = error
        self.progress_bar.hide()
        self.status_label.setText(f"Failed: {str(error)}")
        self._set_status_tone("error")
        self.details_btn.show()
        self._is_active = False
        self.close_btn.show()

        # Automatically pop open the modal on error arrival
        self.show_error_modal()
        traceback.print_exception(type(error), error, error.__traceback__)

    def on_cancel_requested(self) -> None:
        """Called when cancellation is requested, by this toast's 'x' or elsewhere."""
        if not self._is_active:
            return
        self._cancel_requested = True
        self.status_label.setText("Cancelling...")
        self._set_status_tone("warning")
        # Keep the cancel button visible but disabled to indicate request sent
        self.close_btn.setEnabled(False)

    def on_retired(self) -> None:
        """Called when the ProgressController is retired.

        Retirement also happens on normal completion, so it only means
        "cancelling" when a cancellation was actually requested.
        """
        if self._is_active and self._cancel_requested:
            self.on_cancel_requested()

    def _on_close_clicked(self) -> None:
        """Handle clicks on the top-right 'x'. Acts as cancel when active, otherwise closes."""
        if self._is_active:
            # Request cancellation via task signals if available
            if self._task_signals is not None:
                self._task_signals.task_cancel_requested.emit()
            # Update UI immediately
            self.on_cancel_requested()
        else:
            self.fade_out_and_close()

    def on_cancelled(self) -> None:
        """Called when the worker thread reports it has exited due to cancellation."""
        self._is_active = False
        self.progress_bar.hide()
        self.status_label.setText("Cancelled")
        self._set_status_tone("cancelled")
        QTimer.singleShot(1200, self.fade_out_and_close)
