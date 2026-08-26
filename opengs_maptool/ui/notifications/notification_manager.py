from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QVBoxLayout

from opengs_maptool.controllers.task_controller import TaskController, ThreadTask
from opengs_maptool.models.progress_status import ProgressStatus
from opengs_maptool.ui.notifications.task_toast_widget import TaskToastWidget

class NotificationManager(QObject):
    def __init__(self, main_window, task_controller: TaskController, toast_container_layout: QVBoxLayout):
        super().__init__()
        self._main_window = main_window
        self.task_controller = task_controller
        self.toast_container_layout = toast_container_layout

        self._tasks_without_progress = list[tuple[ThreadTask, TaskToastWidget]]()
        self.task_controller.new_task_started.connect(self._on_new_task)
        # Keep track of tasks that haven't started sending progress updates (at least yet)
        # Tasks without progress updates probably shouldn't be shown anyway

    def _on_new_task(self, task: ThreadTask) -> None:
        # When and IF progress actually starts, show widget.
        # However, to not miss the task completed signals, create the widget now already.

        # 1. Create a toast UI widget for this task
        toast = TaskToastWidget(title=task.title, main_window=self._main_window)

        # 2. Wire the 5 signals from TaskSignals to the Toast UI methods
        signals = task.signals

        signals.task_progress_started.connect(toast.on_progress_started)
        signals.task_progress_phase_started.connect(toast.on_phase_started)
        signals.task_progress_updated.connect(toast.on_progress_updated)

        # Notify toast when the controller is retired (owner requested cancel or final retire)
        signals.task_progress_retired.connect(toast.on_retired)
        # Notify toast when worker thread reports cancellation (final outcome)
        signals.task_cancelled.connect(toast.on_cancelled)
        # Notify toast as soon as a cancellation is requested, from anywhere
        signals.task_cancel_requested.connect(toast.on_cancel_requested)

        # Success & Error handlers
        signals.task_successful.connect(toast.on_success)
        signals.task_error.connect(toast.on_error)

        self._tasks_without_progress.append((task, toast))
        task.signals.task_progress_started.connect(
            lambda initial_status: self._on_task_progress_started(toast, task, initial_status)
        )

        # Allow the toast to request cancellation via the task signals
        toast.set_task_signals(task.signals)

    def _on_task_progress_started(self, toast: TaskToastWidget, task: ThreadTask, initial_status: ProgressStatus) -> None:
        # Actually show the toast
        self.toast_container_layout.addWidget(toast)

        # This function runs on the "first" (there should only be one) progress start signal,
        # so we should pass that information to the toast widget, so it doesn't miss it.
        toast.on_progress_started(initial_status)
        # Remove
        self._tasks_without_progress = [(t, w) for t, w in self._tasks_without_progress if t != task]
