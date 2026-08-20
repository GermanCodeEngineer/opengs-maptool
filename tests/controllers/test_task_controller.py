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
