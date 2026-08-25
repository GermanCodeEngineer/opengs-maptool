from __future__ import annotations
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import time
from typing import Iterator, Generator, Protocol, TypeVar
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker
from opengs_maptool.models.progress_status import ProgressStatus
from opengs_maptool.services.logging_service import LOGGING_SERVICE


class TaskCancelledInterrupt(BaseException):
    """Raised in worker code when owner requested task cancellation.

    This inherits from BaseException to distinguish it from normal exceptions
    and to make it easier for worker code to handle/control flow without
    being caught by generic Exception handlers.
    """
    pass

@dataclass(frozen=True, unsafe_hash=True) # avoid manipulation
class ProgressPhaseRef:
    """
    May only be created by ProgressController().add_phase, therefore not included in __all__.
    Should be given to ProgressController().execute_phase() to mark the phase as done.
    """
    _step_weight: int
    _target_completed_steps: int
    _description: str

T = TypeVar("T")  # Generic type variable for track_iteration
class _SizedIterable(Protocol[T]):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...

class ProgressController(QObject):
    """
    API to create and update the status of a progress bar via signals.

    ## Usage in logic thread:
    See the developer document (docs/developer_knowledge/configuring_progress_bars.md)

    ## Usage in UI thread:
    ```
    progress_controller = ProgressController() # uppermost level is created by ui thread
    # ProgressController automatically emits update signals to the UI (pyqt signals)

    # Register signal handlers to update the UI (simplified example):
    progress_controller.task_started.connect(lambda initial_status: ...)
    prorgess_controller.task_phase_started(lambda phase_name, status: ...)
    progress_controller.task_progress_updated.connect(lambda updated_status, increased_steps: ...)
    progress_controller.task_retired.connect(lambda: ...)

    # Start execution
    function_for_my_task(progress_controller) # (this would be run in a different thread though)
    ```
    """


    # Public API for reciever of progress updates including signals and __init__

    task_started = pyqtSignal(ProgressStatus)
    """args: (initial status)"""
    task_phase_started = pyqtSignal(str, ProgressStatus)
    """args: (phase description, current status)"""
    task_progress_updated = pyqtSignal(ProgressStatus, int)
    """args: (updated status, steps completed since last update)"""
    task_retired = pyqtSignal()
    """args: ()"""

    def __init__(self, name: str = "Unnamed"):
        super().__init__()
        self._status = ProgressStatus()
        self._total_configured_steps = 0
        self._phases = list[ProgressPhaseRef]()
        self._has_started_execution = False
        # Cancellation flag protected by a QMutex
        self._has_cancelled_execution = False
        self._cancel_mutex = QMutex()
        self._is_retired = False

        self._name: str = name
        self._logging_name: str = f"ProgressController({self._name})"
        self._phase_start_times = dict[ProgressPhaseRef, float]()
        self._phase_durations = dict[ProgressPhaseRef, float]()
        self._current_phase_description: str | None = None

        self._log("Initialized controller")

    def get_progress_status(self) -> ProgressStatus:
        return copy.copy(self._status)

    def get_progress_quotient(self) -> float:
        return self._status.get_progress_quotient()

    # Public API for sender of progress updates (configuration phase)

    def add_phase(self, step_weight: int, msg: str | None = None) -> ProgressPhaseRef:
        """
        Plan a phase of a task with a step weight. Returns a reference to the phase, which can be used to start it later.
        Usage: see docstring of class.
        """
        if (not isinstance(step_weight, int)) or (step_weight <= 0):
            raise ValueError("step_weight must be a positive integer.")
        if self._has_started_execution:
            raise RuntimeError(
                "Cannot add phases to ProgressController after execution has once been started. "
                "Please add all phases before starting execution or create a new ProgressController."
            )

        self._total_configured_steps += step_weight
        phase_ref = ProgressPhaseRef(
            _step_weight=step_weight,
            _target_completed_steps=self._total_configured_steps,
            _description=msg or f"Phase {len(self._phases) + 1}",
        )
        self._phases.append(phase_ref)
        self._log(
            "Added phase '%s' (weight: %d, cumulative total steps: %d)",
            phase_ref._description,
            step_weight,
            self._total_configured_steps,
        )
        return phase_ref

    # Public API for sender of progress updates (execution phase)

    @contextmanager
    def execute_phase(self, phase_ref: ProgressPhaseRef) -> Generator[SubProgressController]:
        """
        Safely execute a phase of a task with progress updates for the UI and yield a SubProgressController.
        Correctly handles cleanup on exceptions, returns etc.
        Usage: see docstring of class.
        """
        if phase_ref not in self._phases:
            raise ValueError("phase_ref does not belong to this ProgressController.")
        if self._is_retired:
            raise RuntimeError(
                "Cannot execute a phase on a ProgressController that has already been completed or retired because of an interruption."
            )

        self._note_task_started()
        self._note_phase_started(phase_ref)

        start_offset = phase_ref._target_completed_steps - phase_ref._step_weight
        sub_controller = SubProgressController(
            name=f"{self._name}->{phase_ref._description}",
            parent_controller=self,
            step_budget_in_parent=phase_ref._step_weight,
            parent_start_offset=start_offset,
        )

        # Track if this phase block finishes normally
        completed_normally = False
        try:
            yield sub_controller # => Let with-clause run (look up @contextmanager if you don't understand)
            completed_normally = True

        finally:
            if completed_normally:
                # Advance progress to the target step count of this phase
                self._update_progress(completed_steps=phase_ref._target_completed_steps)
                self._note_phase_completed(phase_ref)

                # If this was the last configured phase, retire normally
                if phase_ref is self._phases[-1]:
                    self._retire_progress()
            else:
                # The block raised. Retire immediately so the UI doesn't hang
                # waiting for the remaining phases.
                # NOTE: an early 'return' in the block does NOT land here --
                # @contextmanager resumes the generator normally in that case,
                # so completed_normally is True. ProgressController.retire(),
                # called by the task owner, is what closes out that path.
                self._log("Phase '%s' raised an exception", phase_ref._description)
                self._retire_progress()

    @contextmanager
    def execute_as_only_phase(self):
        """
        Initialize the controller and retire after the block in the with-clause is done.
        Correctly handles cleanup on exceptions, returns etc.
        Usage: see docstring of class.
        """
        if self._phases:
            raise RuntimeError(
                "execute_as_only_phase can only be used on a ProgressController that has no configured phases."
            )

        self._log("Executing single-phase task")
        only_phase = self.add_phase(step_weight=1, msg="Only Phase")
        with self.execute_phase(only_phase):
            yield None # => Let with-clause run


    def retire(self) -> None:
        """Close out the progress bar. Idempotent, safe to call more than once.

        Normally happens automatically when the last configured phase finishes.
        The task owner also calls this when the task function returns, because a
        function may exit early -- e.g. a guard clause returning before its
        remaining phases run -- and would otherwise leave the bar never retired.
        """
        self._retire_progress()

    def track_iteration(self, sequence: _SizedIterable[T]) -> Iterator[T]:
        """
        Wraps a sized iterable (list, tuple, array, etc.) to update progress automatically item-by-item.
        Usage: see docstring of class.
        """
        if self._phases:
            raise RuntimeError(
                "track_iteration can only be used on a ProgressController that has no configured phases."
            )

        if not hasattr(sequence, "__len__"):
            raise TypeError("track_iteration requires a sized collection (e.g. list, tuple, array) with a len().")

        total_items = len(sequence)
        self._log("Tracking iteration task over %d items", total_items)

        # Configure the total steps for this controller instance
        self._total_configured_steps += total_items
        self._note_task_started()

        completed = 0
        try:
            for item in sequence:
                yield item
                completed += 1
                self._update_progress(completed_steps=completed)
        finally:
            if (completed == total_items) or self._has_started_execution:
                self._retire_progress()
        self._note_phase_completed(ProgressPhaseRef(
            _step_weight=total_items, # ignored
            _target_completed_steps=self._total_configured_steps, # ignored
            _description="Iteration",
        ))

    # Helpers

    def _note_task_started(self) -> None:
        if not self._has_started_execution:
            self._status.total_steps = self._total_configured_steps
            self._status.completed_steps = 0
            self._current_phase_description = None

            self._has_started_execution = True
            self._log("Task started with total configured steps: %d", self._total_configured_steps)
            self.task_started.emit(self.get_progress_status())

    def _update_progress(self, completed_steps: int) -> int:
        """
        Sets absolute completed steps. Returns the number of steps increased since the last update.
        """
        # Check for cancellation request before validating/updating progress
        if self.is_cancelled():
            self._log("Cancellation detected while updating progress")
            # ensure UI does not hang waiting for remaining phases
            self._retire_progress()
            raise TaskCancelledInterrupt("Task was cancelled by owner.")

        if (not isinstance(completed_steps, int)) or (completed_steps < 0):
            raise ValueError("completed_steps must be a non-negative integer.")

        if completed_steps > self._status.total_steps:
            raise RuntimeError(
                f"Completed steps ({completed_steps}) cannot exceed total steps ({self._status.total_steps})."
            )

        steps_increased = completed_steps - self._status.completed_steps
        if steps_increased < 0:
            raise ValueError("completed_steps cannot decrease.")

        if steps_increased > 0:
            self._status.completed_steps = completed_steps
            pct = self.get_progress_quotient() * 100
            self._log(
                "Progress updated: %d/%d steps (+%d steps, %.1f%%)",
                completed_steps,
                self._status.total_steps,
                steps_increased,
                pct,
            )
            self.task_progress_updated.emit(self.get_progress_status(), steps_increased)

        return steps_increased

    def _retire_progress(self) -> None:
        if not self._is_retired:
            self._is_retired = True
            self._log("Task retired/completed")
            self.task_retired.emit()

    def _note_phase_started(self, phase_ref: ProgressPhaseRef) -> None:
        self._phase_start_times[phase_ref] = time.perf_counter()
        self._current_phase_description = phase_ref._description
        self._log("Phase started: '%s' (weight: %d)", phase_ref._description, phase_ref._step_weight)
        self.task_phase_started.emit(phase_ref._description, self.get_progress_status())

    def _note_phase_completed(self, phase_ref: ProgressPhaseRef) -> None:
        phase_start_time = self._phase_start_times.get(phase_ref)
        duration = (time.perf_counter() - phase_start_time) if phase_start_time is not None else 0.0
        self._phase_durations[phase_ref] = duration
        self._log("Phase completed: '%s', duration: %.2f seconds", phase_ref._description, duration)

    # Cancellation API
    def cancel(self) -> None:
        """Request cancellation from the owner/UI thread.

        This sets an internal flag protected by a QMutex, emits `task_cancelled`,
        and retires the progress so the UI does not wait for remaining phases.
        """

        with QMutexLocker(self._cancel_mutex):
            if not self._has_cancelled_execution:
                self._has_cancelled_execution = True

        # Retire immediately so UI doesn't hang waiting for worker to hit next update
        self._log("Cancellation requested by owner")
        self._retire_progress()

    def is_cancelled(self) -> bool:
        """Thread-safe check whether cancellation was requested."""
        with QMutexLocker(self._cancel_mutex):
            return self._has_cancelled_execution

    def _log(self, format_str: str, *args) -> None:
        message = f"{self._logging_name}: {format_str % args}"
        LOGGING_SERVICE.loggers.task_progress.send(message)


class SubProgressController(ProgressController):
    """
    A sub-progress controller, that propagates progress updates in a scaled manner to a parent progress controller.
    The sub-progress controller can be used to manage progress for a sub-task that is part of a larger task,
    allowing the parent progress controller to reflect the overall progress of the entire task.
    Nesting SubProgressControllers is possible, but the step budget in the parent should be set to a high value to avoid rounding errors.

    RECOMMENDATION: set step_budget_in_parent to a high value if possible.
    CONVENTION: Every function call with multiple execution steps should get a SubProgressController.
    """
    def __init__(
        self,
        name: str,
        parent_controller: ProgressController,
        step_budget_in_parent: int,
        parent_start_offset: int,
    ):
        super().__init__(name=name)
        self._parent_controller = parent_controller
        self._step_budget_in_parent = step_budget_in_parent
        self._parent_start_offset = parent_start_offset

    # Cancellation: delegate to parent controller so sub-controllers share
    # the same cancellation state and mutex. This avoids duplicating
    # cancel flags and ensures owner-requested cancellation is visible
    # to all nested controllers immediately.
    def cancel(self) -> None:
        self._parent_controller.cancel()

    def is_cancelled(self) -> bool:
        return self._parent_controller.is_cancelled() or super().is_cancelled()

    def _update_progress(self, completed_steps: int) -> int:
        steps_increased = super()._update_progress(completed_steps)

        if self._status.total_steps > 0:
            progress_ratio = completed_steps / self._status.total_steps
            parent_target_steps = self._parent_start_offset + int( # round down
                progress_ratio * self._step_budget_in_parent
            )
            self._parent_controller._update_progress(parent_target_steps)

        return steps_increased


__all__ = ["ProgressController", "SubProgressController"]
