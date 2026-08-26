# GCE-TODO: update tests according to implementation changes
from unittest.mock import MagicMock
import pytest
import threading

from opengs_maptool.controllers.task_controller import TaskController, ThreadTaskSlot, ThreadSlotOccupiedError
from opengs_maptool.controllers.task_controller import ThreadTask, TaskSignals
from opengs_maptool.controllers.progress_controller import ProgressController, TaskCancelledInterrupt
from unittest.mock import patch


class _SyncThreadPool:
    """Fake thread pool that executes runnables synchronously for testing."""
    def start(self, runnable):
        runnable.run()


def test_start_task_success_and_slot_freed(monkeypatch):
    # Patch global thread pool to run tasks synchronously
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    started = []
    controller.new_task_started.connect(lambda task: started.append(task))

    def worker(progress_controller):
        return "ok"

    task = controller.start_task(worker, title="t1", slot=ThreadTaskSlot.change_density_image, pos_args=[], kw_args={})
    assert started, "new_task_started should have been emitted"
    # After synchronous run the slot should have been freed
    assert not controller.is_thread_slot_occupied(ThreadTaskSlot.change_density_image)


def test_start_task_error_frees_slot(monkeypatch):
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()

    def worker(progress_controller):
        raise RuntimeError("boom")

    controller.start_task(worker, title="t_err", slot=ThreadTaskSlot.generate_province_map, pos_args=[], kw_args={})
    # slot freed even on error
    assert not controller.is_thread_slot_occupied(ThreadTaskSlot.generate_province_map)


def test_start_task_slot_occupied_raises(monkeypatch):
    # Use a pool that does not run tasks so the slot remains occupied
    class _NoRunThreadPool:
        def start(self, runnable):
            # intentionally do not call runnable.run() to simulate long-running task
            return

    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _NoRunThreadPool(),
    )

    controller = TaskController()

    def worker(progress_controller):
        return None

    # Start first task
    controller.start_task(worker, title="t1", slot=ThreadTaskSlot.generate_territory_map, pos_args=[], kw_args={})

    # Trying to start another in the same slot should raise
    with pytest.raises(ThreadSlotOccupiedError):
        controller.start_task(worker, title="t2", slot=ThreadTaskSlot.generate_territory_map, pos_args=[], kw_args={})


def test_threadtask_run_emits_task_cancelled_on_TaskCancelledInterrupt():
    # Create a ThreadTask whose worker raises TaskCancelledInterrupt
    pc = ProgressController()

    def worker(progress_controller):
        raise TaskCancelledInterrupt()

    task = ThreadTask(title="t", slot=ThreadTaskSlot.change_density_image, progress_controller=pc, function=worker)
    cancelled = MagicMock()
    task.signals.task_cancelled.connect(cancelled)

    # Run synchronously
    task.run()
    cancelled.assert_called_once()


def test_threadtask_forwards_progress_signals_to_tasksignals():
    pc = ProgressController()

    def worker(progress_controller):
        return None

    task = ThreadTask(title="t", slot=ThreadTaskSlot.change_density_image, progress_controller=pc, function=worker)

    started_mock = MagicMock()
    task.signals.task_progress_started.connect(started_mock)

    # Emit progress_controller.task_started and ensure it is forwarded
    pc.task_started.emit(pc.get_progress_status())
    started_mock.assert_called_once()


def test_cancel_all_and_wait_no_tasks_returns_true():
    controller = TaskController()
    assert controller.cancel_all_and_wait() is True


def test_cancel_all_and_wait_with_occupied_slot_times_out(monkeypatch):
    # Use a pool that does not run tasks so the slot remains occupied
    class _NoRunThreadPool:
        def start(self, runnable):
            return

    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _NoRunThreadPool(),
    )

    # Replace QEventLoop and QTimer with test-friendly fakes so the test can
    # exercise the timeout path without requiring a QApplication event loop.
    class _FakeEventLoop:
        def __init__(self):
            self._evt = threading.Event()

        def exec(self):
            # block until quit() is called or timer fires
            self._evt.wait()

        def quit(self):
            self._evt.set()

    class _FakeTimeout:
        def __init__(self):
            self._cb = None

        def connect(self, cb):
            self._cb = cb

        def fire(self):
            if self._cb:
                self._cb()

    class _FakeTimer:
        def __init__(self):
            self._timeout = _FakeTimeout()
            self._timer = None

        @property
        def timeout(self):
            return self._timeout

        def setSingleShot(self, v):
            self._single = v

        def start(self, ms):
            # schedule the timeout to call after ms milliseconds
            self._timer = threading.Timer(ms / 1000.0, self._timeout.fire)
            self._timer.start()

        def stop(self):
            if self._timer:
                self._timer.cancel()

    monkeypatch.setattr('opengs_maptool.controllers.task_controller.QEventLoop', _FakeEventLoop)
    monkeypatch.setattr('opengs_maptool.controllers.task_controller.QTimer', _FakeTimer)

    controller = TaskController()

    def worker(progress_controller):
        return None

    controller.start_task(worker, title="t1", slot=ThreadTaskSlot.generate_territory_map, pos_args=[], kw_args={})

    # With a short timeout, the waiting should time out and return False
    assert controller.cancel_all_and_wait(max_wait_ms=20) is False


def test_new_task_started_is_emitted_before_the_task_runs(monkeypatch):
    """Listeners must be able to connect to the terminal signals before the worker runs.

    A fast task can otherwise finish and emit task_successful while nobody is
    listening yet, which left the progress toast stuck on its last intermediate
    state. _SyncThreadPool runs the task inside start(), making that race
    deterministic.
    """
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    received = []

    # Mimic NotificationManager: connect to the task's signals on new_task_started
    controller.new_task_started.connect(
        lambda task: task.signals.task_successful.connect(lambda result: received.append(result))
    )

    def worker(progress_controller):
        return "ok"

    controller.start_task(worker, title="fast", slot=ThreadTaskSlot.change_density_image, pos_args=[], kw_args={})

    assert received == ["ok"], "task_successful was emitted before listeners could connect"


def test_task_function_returning_early_still_retires_progress(monkeypatch):
    """A guard clause may return before the remaining phases run.

    @contextmanager resumes the generator normally on an early 'return', so
    execute_phase cannot tell that case apart from a completed block. Without
    the owner retiring, the progress bar would never be closed out.
    """
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    retired = MagicMock()

    def worker(progress_controller):
        first = progress_controller.add_phase(1, "first")
        progress_controller.add_phase(1, "never reached")
        progress_controller.task_retired.connect(retired)
        with progress_controller.execute_phase(first):
            return  # guard clause: nothing to do

    task = controller.start_task(
        worker, title="early", slot=ThreadTaskSlot.change_density_image, pos_args=[], kw_args={}
    )

    retired.assert_called_once()
    assert task.progress_controller._is_retired


def test_progress_controller_is_retired_exactly_once_on_normal_completion(monkeypatch):
    """The owner's retire() must not emit a second task_retired after the last phase."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    retired = MagicMock()

    def worker(progress_controller):
        phases = [progress_controller.add_phase(1, f"p{i}") for i in range(3)]
        progress_controller.task_retired.connect(retired)
        for phase in phases:
            with progress_controller.execute_phase(phase):
                pass

    controller.start_task(
        worker, title="normal", slot=ThreadTaskSlot.change_density_image, pos_args=[], kw_args={}
    )

    retired.assert_called_once()
