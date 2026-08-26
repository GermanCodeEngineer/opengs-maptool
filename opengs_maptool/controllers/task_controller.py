from __future__ import annotations
from enum import Enum, auto
from typing import Any, Callable
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot, QEventLoop, QTimer

from opengs_maptool.controllers.progress_controller import ProgressController, TaskCancelledInterrupt
from opengs_maptool.models.progress_status import ProgressStatus

class ThreadTaskSlot(Enum):
    change_density_image = auto()
    generate_territory_map = auto()
    generate_province_map = auto()
    execute_command = auto()

class TaskSignals(QObject):
    task_error = pyqtSignal(BaseException)
    task_successful = pyqtSignal(object)
    """args: (return value)"""

    # Emitted to request cancellation of the background task
    task_cancel_requested = pyqtSignal()
    # Emitted when the worker thread actually exits due to a cancellation
    task_cancelled = pyqtSignal()

    # also see signal TaskController.new_task_started and others

    # Optional convenience forwarded signals from ProgressController
    task_progress_started = pyqtSignal(ProgressStatus)
    """args: (initial status)"""
    task_progress_phase_started = pyqtSignal(str, ProgressStatus)
    """args: (phase description, current status)"""
    task_progress_updated = pyqtSignal(ProgressStatus, int)
    """args: (updated status, steps completed since last update)"""
    task_progress_retired = pyqtSignal()
    """args: ()"""

class ThreadTask(QRunnable):
    def __init__(self,
        title: str,
        slot: ThreadTaskSlot,
        progress_controller: ProgressController | None,

        function: Callable[..., Any],
        pos_args: tuple[Any, ...] = (),
        kw_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        # Intended for public
        self.title = title
        self.slot = slot
        self.progress_controller = progress_controller
        self.signals = TaskSignals()
        # Forward progress signals from ProgressController to TaskSignals
        if self.progress_controller:
            self.progress_controller.task_started.connect(self.signals.task_progress_started)
            self.progress_controller.task_phase_started.connect(self.signals.task_progress_phase_started)
            self.progress_controller.task_progress_updated.connect(self.signals.task_progress_updated)
            self.progress_controller.task_retired.connect(self.signals.task_progress_retired)
            # Connect cancel request signal to the ProgressController.cancel method so
            # the owner can request cancellation via the signal.
            self.signals.task_cancel_requested.connect(self.progress_controller.cancel)

        self._function = function
        self._args = pos_args
        self._kwargs = kw_args or {}
        if self.progress_controller:
            self._kwargs["progress_controller"] = self.progress_controller

    @pyqtSlot()
    def run(self) -> None:
        try:
            try:
                result = self._function(*self._args, **self._kwargs)
            finally:
                # The task function may return without executing every phase it
                # configured (e.g. a guard clause). Retire here so the progress
                # bar is always closed out, before the terminal signal goes out.
                self.progress_controller.retire()
        except TaskCancelledInterrupt:
            # Worker was cancelled; emit a dedicated cancelled signal so the UI
            # can treat this case separately from errors.
            self.signals.task_cancelled.emit()
        except Exception as error:
            self.signals.task_error.emit(error)
        else:
            self.signals.task_successful.emit(result)

class TaskController(QObject):
    new_task_started = pyqtSignal(ThreadTask)
    thread_task_slot_occupied = pyqtSignal(ThreadTaskSlot)
    thread_task_slot_freed = pyqtSignal(ThreadTaskSlot)

    def __init__(self):
        super().__init__()
        self._thread_pool = QThreadPool.globalInstance()
        self._slot_tasks: dict[ThreadTaskSlot, ThreadTask] = {}

    def cancel_all_and_wait(self, max_wait_ms: int | None = None) -> bool:
        """Request cancellation for all running tasks and wait until all slots are freed.

        Returns True if all tasks terminated before the timeout, False otherwise.
        If `max_wait_ms` is None the method will wait indefinitely (may block the UI).
        """
        # Snapshot current tasks
        tasks = list(self._slot_tasks.values())
        if not tasks:
            return True

        # Emit cancel requests via each task's TaskSignals so queued-slot semantics apply
        for task in tasks:
            task.signals.task_cancel_requested.emit()

        remaining = len(tasks)
        loop = QEventLoop()

        def on_freed(slot: ThreadTaskSlot) -> None:
            nonlocal remaining
            remaining -= 1
            if remaining <= 0:
                loop.quit()

        self.thread_task_slot_freed.connect(on_freed)

        timer = None
        if max_wait_ms is not None:
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(loop.quit)
            timer.start(max_wait_ms)

        loop.exec()

        if timer is not None:
            timer.stop()
        self.thread_task_slot_freed.disconnect(on_freed)

        return remaining <= 0

    def is_thread_slot_occupied(self, slot: ThreadTaskSlot) -> bool:
        """Check if a specific thread slot is currently occupied by a running task."""
        occupying_task = self._slot_tasks.get(slot)
        return occupying_task is not None

    def start_task(self,
            function: Callable[..., Any], title: str, slot: ThreadTaskSlot,
            provide_progress_controller: bool, pos_args: list[Any], kw_args: dict[str, Any],
            before_start_callback: Callable[[ThreadTask], None] | None,
        ) -> ThreadTask:
        """
        Start a task in a background thread and get a variously useful ThreadTask object back.
        Raises:
            ThreadSlotOccupiedError: If a task is already running in the specified slot.
        """
        if self.is_thread_slot_occupied(slot):
            occupying_task = self._slot_tasks[slot]
            raise ThreadSlotOccupiedError(f"Cannot start task in slot {slot.name}, because the task {occupying_task.title} is already running.")

        # Create task
        task = ThreadTask(
            title=title,
            slot=slot,
            progress_controller=(ProgressController(
                name=slot.name,
            ) if provide_progress_controller else None),

            function=function,
            pos_args=pos_args,
            kw_args=kw_args,
        )
        # Occupy the slot
        self._set_slot(slot, task)

        # Announce the task *before* starting it. Listeners (e.g. the toast
        # notifications) connect to the terminal signals inside this emit, and a
        # fast task can otherwise finish and emit task_successful before anyone
        # is listening, leaving the UI stuck on its last intermediate state.
        if before_start_callback is not None:
            before_start_callback(task)

        self.new_task_started.emit(task)
        self._thread_pool.start(task)
        return task

    def _set_slot(self, slot: ThreadTaskSlot, task: ThreadTask) -> None:
        self._slot_tasks[slot] = task
        task.signals.task_successful.connect(lambda result: self._free_slot(slot))
        task.signals.task_error.connect(lambda error: self._free_slot(slot))
        task.signals.task_cancelled.connect(lambda: self._free_slot(slot))
        self.thread_task_slot_occupied.emit(slot)

    def _free_slot(self, slot: ThreadTaskSlot) -> None:
        if self.is_thread_slot_occupied(slot):
            del self._slot_tasks[slot]
            self.thread_task_slot_freed.emit(slot)

class ThreadSlotOccupiedError(Exception):
    """Raised when attempting to start a task while another background task is active."""
