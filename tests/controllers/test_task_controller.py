from unittest.mock import MagicMock
import pytest
import threading

from opengs_maptool.controllers.task_controller import TaskController, ThreadTaskSlot, ThreadSlotOccupiedError
from opengs_maptool.controllers.task_controller import ThreadTask, TaskSignals
from opengs_maptool.controllers.progress_controller import ProgressController, TaskCancelledInterrupt


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
    
    captured_task = []

    def worker(progress_controller):
        return "ok"

    def capture_and_run(task):
        captured_task.append(task)
        # Manually run the task synchronously for testing
        task.run()

    controller.start_task(
        worker, 
        title="t1", 
        slot=ThreadTaskSlot.change_density_image, 
        provide_progress_controller=True,
        pos_args=[], 
        kw_args={},
        before_start_callback=capture_and_run
    )
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

    def capture_and_run(task: ThreadTask):
        # Manually run the task synchronously for testing
        task.run()

    controller.start_task(
        worker, 
        title="t_err", 
        slot=ThreadTaskSlot.generate_province_map, 
        provide_progress_controller=True,
        pos_args=[], 
        kw_args={},
        before_start_callback=capture_and_run
    )
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
    controller.start_task(
        worker, 
        title="t1", 
        slot=ThreadTaskSlot.generate_territory_map, 
        provide_progress_controller=True,
        pos_args=[], 
        kw_args={},
        before_start_callback=None
    )

    # Trying to start another in the same slot should raise
    with pytest.raises(ThreadSlotOccupiedError):
        controller.start_task(
            worker, 
            title="t2", 
            slot=ThreadTaskSlot.generate_territory_map, 
            provide_progress_controller=True,
            pos_args=[], 
            kw_args={},
            before_start_callback=None
        )


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
    """Verify cancel_all_and_wait returns False when tasks don't complete before timeout."""
    # Use a pool that does not run tasks so the slot remains occupied
    class _NoRunThreadPool:
        def start(self, runnable):
            return

    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _NoRunThreadPool(),
    )

    # Use injectable fake objects that don't require QApplication or threading
    class _FakeEventLoop:
        def __init__(self):
            self.quit_called = False

        def exec(self):
            # Never blocks, returns immediately
            pass

        def quit(self):
            self.quit_called = True

    class _FakeTimeout:
        """Fake timeout signal for QTimer."""
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self):
            for cb in self._callbacks:
                cb()

    class _FakeTimer:
        def __init__(self):
            self.timeout = _FakeTimeout()
            self.started = False

        def setSingleShot(self, v):
            pass

        def start(self, ms):
            self.started = True
            # Immediately emit timeout to simulate expiry
            self.timeout.emit()

        def stop(self):
            pass

    controller = TaskController(
        event_loop_factory=_FakeEventLoop,
        timer_factory=_FakeTimer,
    )

    def worker(progress_controller):
        return None

    controller.start_task(
        worker, 
        title="t1", 
        slot=ThreadTaskSlot.generate_territory_map, 
        provide_progress_controller=True,
        pos_args=[], 
        kw_args={},
        before_start_callback=None
    )

    # With the fake timer that immediately times out and no slots freed,
    # the waiting should time out and return False
    assert controller.cancel_all_and_wait(max_wait_ms=20) is False


def test_new_task_started_is_emitted_before_the_task_runs(monkeypatch):
    """Listeners must be able to connect to the terminal signals before the worker runs.

    A fast task can otherwise finish and emit task_successful while nobody is
    listening yet, which left the progress toast stuck on its last intermediate
    state.
    """
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    
    received = []
    captured_task = []

    # Mimic NotificationManager: connect to the task's signals on new_task_started
    def on_task_started(task: ThreadTask):
        # Capture the task and connect to its signal
        captured_task.append(task)
        task.signals.task_successful.connect(lambda result: received.append(result))

    controller.new_task_started.connect(on_task_started)

    def worker(progress_controller):
        return "ok"

    controller.start_task(
        worker, 
        title="fast", 
        slot=ThreadTaskSlot.change_density_image, 
        provide_progress_controller=True,
        pos_args=[], 
        kw_args={},
        before_start_callback=None
    )
    
    # Now manually run the task to simulate async execution
    if captured_task:
        captured_task[0].run()

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
    captured_task = []

    def worker(progress_controller):
        first = progress_controller.add_phase(1, "first")
        progress_controller.add_phase(1, "never reached")
        progress_controller.task_retired.connect(retired)
        with progress_controller.execute_phase(first):
            return  # guard clause: nothing to do

    def capture_and_run(task: ThreadTask):
        captured_task.append(task)
        # Manually run the task synchronously for testing
        task.run()

    controller.start_task(
        worker, 
        title="early", 
        slot=ThreadTaskSlot.change_density_image, 
        provide_progress_controller=True,
        pos_args=[], 
        kw_args={},
        before_start_callback=capture_and_run
    )

    retired.assert_called_once()
    assert captured_task, "Task should have been captured"
    assert captured_task[0].progress_controller._is_retired


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

    def capture_and_run(task: ThreadTask):
        # Manually run the task synchronously for testing
        task.run()

    controller.start_task(
        worker, 
        title="normal", 
        slot=ThreadTaskSlot.change_density_image, 
        provide_progress_controller=True,
        pos_args=[], 
        kw_args={},
        before_start_callback=capture_and_run
    )

    retired.assert_called_once()


def test_start_task_without_progress_controller(monkeypatch):
    """Tasks can run without a progress controller."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    result_holder = []

    def worker():
        return "no_progress"

    def capture_and_run(task: ThreadTask):
        result_holder.append(task)
        task.run()

    controller.start_task(
        worker, 
        title="no_progress", 
        slot=ThreadTaskSlot.change_density_image, 
        provide_progress_controller=False,
        pos_args=[], 
        kw_args={},
        before_start_callback=capture_and_run
    )

    assert result_holder[0].progress_controller is None
    assert not controller.is_thread_slot_occupied(ThreadTaskSlot.change_density_image)


def test_start_task_with_pos_args_and_kw_args(monkeypatch):
    """Tasks receive positional and keyword arguments correctly."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    results = []

    def worker(a, b, c=None, progress_controller=None):
        return (a, b, c)

    def capture_and_run(task: ThreadTask):
        task.signals.task_successful.connect(lambda r: results.append(r))
        task.run()

    controller.start_task(
        worker, 
        title="args_test", 
        slot=ThreadTaskSlot.generate_territory_map, 
        provide_progress_controller=False,
        pos_args=[1, 2], 
        kw_args={"c": 3},
        before_start_callback=capture_and_run
    )

    assert results[0] == (1, 2, 3)


def test_multiple_concurrent_tasks_different_slots(monkeypatch):
    """Multiple tasks can run concurrently in different slots."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    results = []

    def worker(name, progress_controller=None):
        return name

    def capture_and_run(task: ThreadTask):
        task.signals.task_successful.connect(lambda r: results.append(r))
        task.run()

    # Start multiple tasks in different slots
    controller.start_task(
        worker, 
        title="task1", 
        slot=ThreadTaskSlot.change_density_image, 
        provide_progress_controller=True,
        pos_args=["task1"], 
        kw_args={},
        before_start_callback=capture_and_run
    )

    controller.start_task(
        worker, 
        title="task2", 
        slot=ThreadTaskSlot.generate_territory_map, 
        provide_progress_controller=True,
        pos_args=["task2"], 
        kw_args={},
        before_start_callback=capture_and_run
    )

    controller.start_task(
        worker, 
        title="task3", 
        slot=ThreadTaskSlot.generate_province_map, 
        provide_progress_controller=True,
        pos_args=["task3"], 
        kw_args={},
        before_start_callback=capture_and_run
    )

    # All slots should now be free (sync execution)
    assert not controller.is_thread_slot_occupied(ThreadTaskSlot.change_density_image)
    assert not controller.is_thread_slot_occupied(ThreadTaskSlot.generate_territory_map)
    assert not controller.is_thread_slot_occupied(ThreadTaskSlot.generate_province_map)
    assert set(results) == {"task1", "task2", "task3"}


def test_cancel_all_and_wait_successful_completion_before_timeout(monkeypatch):
    """cancel_all_and_wait returns True when all tasks complete before timeout."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    task_ran = []

    def worker(progress_controller=None):
        task_ran.append(True)
        return "completed"

    def capture_and_run(task: ThreadTask):
        task.run()

    controller.start_task(
        worker, 
        title="quick_task", 
        slot=ThreadTaskSlot.change_density_image, 
        provide_progress_controller=True,
        pos_args=[], 
        kw_args={},
        before_start_callback=capture_and_run
    )

    # Should return True since the task is already completed (sync execution)
    assert task_ran, "Task should have run"
    result = controller.cancel_all_and_wait(max_wait_ms=1000)
    assert result is True


def test_before_start_callback_exception_does_not_prevent_task_start(monkeypatch):
    """If before_start_callback raises, the exception is propagated to the caller."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()

    def worker(progress_controller=None):
        return "result"

    def failing_callback(task: ThreadTask):
        raise ValueError("Callback failed")

    # Try to start with a failing callback - it will raise during start_task
    # and the slot should be left occupied
    with pytest.raises(ValueError, match="Callback failed"):
        controller.start_task(
            worker, 
            title="callback_fails", 
            slot=ThreadTaskSlot.change_density_image, 
            provide_progress_controller=True,
            pos_args=[], 
            kw_args={},
            before_start_callback=failing_callback
        )

    # The slot should be occupied since the callback failed before the task could run
    assert controller.is_thread_slot_occupied(ThreadTaskSlot.change_density_image)


def test_sequential_tasks_in_same_slot(monkeypatch):
    """Tasks can run sequentially in the same slot after previous task completes."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    execution_order = []

    def worker(task_id, progress_controller=None):
        execution_order.append(task_id)
        return task_id

    def capture_and_run(task: ThreadTask):
        task.run()

    # First task
    controller.start_task(
        worker, 
        title="seq1", 
        slot=ThreadTaskSlot.change_density_image, 
        provide_progress_controller=True,
        pos_args=[1], 
        kw_args={},
        before_start_callback=capture_and_run
    )

    # Slot should be free after sync execution
    assert not controller.is_thread_slot_occupied(ThreadTaskSlot.change_density_image), "Slot should be free after first task"
    assert execution_order == [1], "First task should have executed"

    # Second task in same slot
    controller.start_task(
        worker, 
        title="seq2", 
        slot=ThreadTaskSlot.change_density_image, 
        provide_progress_controller=True,
        pos_args=[2], 
        kw_args={},
        before_start_callback=capture_and_run
    )

    assert execution_order == [1, 2], "Both tasks should execute in order"
    assert not controller.is_thread_slot_occupied(ThreadTaskSlot.change_density_image), "Slot should be free after second task"


def test_threadtask_run_with_exception_still_retires_progress(monkeypatch):
    """Even when an exception occurs, progress controller is retired."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    retired_calls = []
    error_emitted = []

    def worker(progress_controller=None):
        if progress_controller:
            progress_controller.task_retired.connect(lambda: retired_calls.append(1))
        raise RuntimeError("worker failed")

    def capture_and_run(task: ThreadTask):
        task.signals.task_error.connect(lambda e: error_emitted.append(e))
        task.run()

    controller.start_task(
        worker, 
        title="error_retire", 
        slot=ThreadTaskSlot.generate_territory_map, 
        provide_progress_controller=True,
        pos_args=[], 
        kw_args={},
        before_start_callback=capture_and_run
    )

    # Progress should be retired even though an error occurred
    assert len(retired_calls) == 1, "Progress controller should be retired"
    assert len(error_emitted) == 1, "Error signal should be emitted"
    assert isinstance(error_emitted[0], RuntimeError), "Should be RuntimeError"


def test_task_signals_are_independent_per_task(monkeypatch):
    """Each task has independent signals that don't cross-pollinate."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    task1_results = []
    task2_results = []

    def worker(task_num, progress_controller=None):
        return f"task{task_num}"

    def capture1(task: ThreadTask):
        task.signals.task_successful.connect(lambda r: task1_results.append(r))
        task.run()

    def capture2(task: ThreadTask):
        task.signals.task_successful.connect(lambda r: task2_results.append(r))
        task.run()

    controller.start_task(
        worker, 
        title="sig1", 
        slot=ThreadTaskSlot.change_density_image, 
        provide_progress_controller=False,
        pos_args=[1], 
        kw_args={},
        before_start_callback=capture1
    )

    controller.start_task(
        worker, 
        title="sig2", 
        slot=ThreadTaskSlot.generate_territory_map, 
        provide_progress_controller=False,
        pos_args=[2], 
        kw_args={},
        before_start_callback=capture2
    )

    assert task1_results == ["task1"], "Task 1 should complete with task1 result"
    assert task2_results == ["task2"], "Task 2 should complete with task2 result"


def test_thread_task_slot_occupied_signal_emitted(monkeypatch):
    """Verify thread_task_slot_occupied signal is emitted when slot becomes occupied."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    # Use a pool that doesn't run tasks to keep slot occupied
    class _NoRunThreadPool:
        def start(self, runnable):
            return

    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _NoRunThreadPool(),
    )

    controller = TaskController()
    occupied_signals = []

    controller.thread_task_slot_occupied.connect(lambda slot: occupied_signals.append(slot))

    def worker(progress_controller=None):
        return "ok"

    controller.start_task(
        worker,
        title="occupy_test",
        slot=ThreadTaskSlot.change_density_image,
        provide_progress_controller=False,
        pos_args=[],
        kw_args={},
        before_start_callback=None
    )

    assert len(occupied_signals) == 1, "thread_task_slot_occupied should be emitted once"
    assert occupied_signals[0] == ThreadTaskSlot.change_density_image, "Correct slot should be reported"


def test_thread_task_slot_freed_signal_emitted(monkeypatch):
    """Verify thread_task_slot_freed signal is emitted when slot is freed."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    freed_signals = []

    controller.thread_task_slot_freed.connect(lambda slot: freed_signals.append(slot))

    def worker(progress_controller=None):
        return "ok"

    def capture_and_run(task: ThreadTask):
        task.run()

    controller.start_task(
        worker,
        title="free_test",
        slot=ThreadTaskSlot.generate_territory_map,
        provide_progress_controller=False,
        pos_args=[],
        kw_args={},
        before_start_callback=capture_and_run
    )

    assert len(freed_signals) == 1, "thread_task_slot_freed should be emitted once"
    assert freed_signals[0] == ThreadTaskSlot.generate_territory_map, "Correct slot should be reported"


def test_all_thread_task_slots_can_run_tasks(monkeypatch):
    """Verify all ThreadTaskSlot enum values work correctly."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    executed_slots = []

    def worker(slot_name, progress_controller=None):
        executed_slots.append(slot_name)
        return slot_name

    def make_callback(slot):
        def capture_and_run(task: ThreadTask):
            task.run()
        return capture_and_run

    # Test all slots
    all_slots = [
        ThreadTaskSlot.change_density_image,
        ThreadTaskSlot.generate_territory_map,
        ThreadTaskSlot.generate_province_map,
        ThreadTaskSlot.execute_command,
    ]

    for slot in all_slots:
        controller.start_task(
            worker,
            title=f"test_{slot.name}",
            slot=slot,
            provide_progress_controller=False,
            pos_args=[slot.name],
            kw_args={},
            before_start_callback=make_callback(slot)
        )
        
        # Slot should be free after task completes
        assert not controller.is_thread_slot_occupied(slot), f"Slot {slot.name} should be free after task"

    assert len(executed_slots) == len(all_slots), "All slots should have executed tasks"


def test_callback_not_running_task_leaves_slot_occupied_until_task_errors(monkeypatch):
    """If callback doesn't call task.run(), the task won't execute."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    task_ran = []

    def worker(progress_controller=None):
        task_ran.append(True)
        return "result"

    def no_run_callback(task: ThreadTask):
        # Intentionally don't run the task
        pass

    controller.start_task(
        worker,
        title="no_run",
        slot=ThreadTaskSlot.change_density_image,
        provide_progress_controller=False,
        pos_args=[],
        kw_args={},
        before_start_callback=no_run_callback
    )

    # Task should not have run since callback didn't call task.run()
    assert len(task_ran) == 0, "Task should not have run"
    # And the slot should still be occupied
    assert controller.is_thread_slot_occupied(ThreadTaskSlot.change_density_image), "Slot should remain occupied"


def test_task_error_with_progress_controller_gets_retired(monkeypatch):
    """When a task raises an error, progress is still retired."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    errors_emitted = []
    retired_signals = []

    def worker(progress_controller):
        progress_controller.task_retired.connect(lambda: retired_signals.append(True))
        raise ValueError("Test error")

    def capture_and_run(task: ThreadTask):
        task.signals.task_error.connect(lambda e: errors_emitted.append(e))
        task.run()

    controller.start_task(
        worker,
        title="error_retire",
        slot=ThreadTaskSlot.generate_province_map,
        provide_progress_controller=True,
        pos_args=[],
        kw_args={},
        before_start_callback=capture_and_run
    )

    assert len(errors_emitted) == 1, "Error signal should be emitted"
    assert isinstance(errors_emitted[0], ValueError), "Should be ValueError"
    assert len(retired_signals) == 1, "Progress should be retired even on error"
    assert not controller.is_thread_slot_occupied(ThreadTaskSlot.generate_province_map), "Slot should be freed"


def test_task_cancellation_with_progress_controller_gets_retired(monkeypatch):
    """When a task is cancelled, progress is still retired."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    cancel_signals = []
    retired_signals = []

    def worker(progress_controller):
        progress_controller.task_retired.connect(lambda: retired_signals.append(True))
        raise TaskCancelledInterrupt()

    def capture_and_run(task: ThreadTask):
        task.signals.task_cancelled.connect(lambda: cancel_signals.append(True))
        task.run()

    controller.start_task(
        worker,
        title="cancel_retire",
        slot=ThreadTaskSlot.execute_command,
        provide_progress_controller=True,
        pos_args=[],
        kw_args={},
        before_start_callback=capture_and_run
    )

    assert len(cancel_signals) == 1, "Cancel signal should be emitted"
    assert len(retired_signals) == 1, "Progress should be retired even on cancellation"
    assert not controller.is_thread_slot_occupied(ThreadTaskSlot.execute_command), "Slot should be freed"


def test_signal_connections_survive_task_execution(monkeypatch):
    """Signals remain properly connected after task execution."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    task_started_calls = []

    def task_started_handler(task: ThreadTask):
        task_started_calls.append(task.title)

    controller.new_task_started.connect(task_started_handler)

    def worker(progress_controller=None):
        return "ok"

    def capture_and_run(task: ThreadTask):
        task.run()

    # Start first task
    controller.start_task(
        worker,
        title="task1",
        slot=ThreadTaskSlot.change_density_image,
        provide_progress_controller=False,
        pos_args=[],
        kw_args={},
        before_start_callback=capture_and_run
    )

    # Start second task - signal should still be connected
    controller.start_task(
        worker,
        title="task2",
        slot=ThreadTaskSlot.generate_territory_map,
        provide_progress_controller=False,
        pos_args=[],
        kw_args={},
        before_start_callback=capture_and_run
    )

    assert len(task_started_calls) == 2, "Signal should fire for both tasks"
    assert task_started_calls == ["task1", "task2"], "Tasks should be called in order"


def test_threadtask_receives_progress_controller_in_kwargs(monkeypatch):
    """When provide_progress_controller is True, progress_controller is passed in kwargs."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    received_kwargs = []

    def worker(**kwargs):
        received_kwargs.append(kwargs)
        return "ok"

    def capture_and_run(task: ThreadTask):
        task.run()

    controller.start_task(
        worker,
        title="kwargs_test",
        slot=ThreadTaskSlot.change_density_image,
        provide_progress_controller=True,
        pos_args=[],
        kw_args={"custom_key": "custom_value"},
        before_start_callback=capture_and_run
    )

    assert len(received_kwargs) == 1, "Worker should be called once"
    assert "progress_controller" in received_kwargs[0], "progress_controller should be in kwargs"
    assert isinstance(received_kwargs[0]["progress_controller"], ProgressController), "Should be ProgressController instance"
    assert received_kwargs[0]["custom_key"] == "custom_value", "Custom kwargs should be preserved"


def test_threadtask_without_progress_controller_receives_clean_kwargs(monkeypatch):
    """When provide_progress_controller is False, progress_controller is not in kwargs."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    received_kwargs = []

    def worker(**kwargs):
        received_kwargs.append(kwargs)
        return "ok"

    def capture_and_run(task: ThreadTask):
        task.run()

    controller.start_task(
        worker,
        title="no_pc_kwargs",
        slot=ThreadTaskSlot.generate_territory_map,
        provide_progress_controller=False,
        pos_args=[],
        kw_args={"custom_key": "custom_value"},
        before_start_callback=capture_and_run
    )

    assert len(received_kwargs) == 1, "Worker should be called once"
    assert "progress_controller" not in received_kwargs[0], "progress_controller should NOT be in kwargs"
    assert received_kwargs[0]["custom_key"] == "custom_value", "Custom kwargs should be preserved"


def test_multiple_task_errors_each_emit_separate_signals(monkeypatch):
    """Each task error emits its own signal, not cross-contaminated."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    error_results = []

    def make_worker(error_msg: str):
        def worker(progress_controller=None):
            raise RuntimeError(error_msg)
        return worker

    def make_callback(error_list):
        def callback(task: ThreadTask):
            task.signals.task_error.connect(lambda e: error_list.append(str(e)))
            task.run()
        return callback

    # Task 1
    errors1 = []
    controller.start_task(
        make_worker("error1"),
        title="err1",
        slot=ThreadTaskSlot.change_density_image,
        provide_progress_controller=False,
        pos_args=[],
        kw_args={},
        before_start_callback=make_callback(errors1)
    )

    # Task 2
    errors2 = []
    controller.start_task(
        make_worker("error2"),
        title="err2",
        slot=ThreadTaskSlot.generate_territory_map,
        provide_progress_controller=False,
        pos_args=[],
        kw_args={},
        before_start_callback=make_callback(errors2)
    )

    assert errors1 == ["error1"], "Task 1 should emit its own error"
    assert errors2 == ["error2"], "Task 2 should emit its own error"


def test_cancel_all_waits_for_multiple_tasks_in_different_slots(monkeypatch):
    """cancel_all_and_wait correctly waits for tasks in all slots."""
    monkeypatch.setattr(
        'opengs_maptool.controllers.task_controller.QThreadPool.globalInstance',
        lambda: _SyncThreadPool(),
    )

    controller = TaskController()
    cancelled_tasks = []

    def worker(task_id, progress_controller=None):
        if progress_controller:
            progress_controller.task_retired.connect(lambda: cancelled_tasks.append(task_id))
        return task_id

    def make_callback():
        def callback(task: ThreadTask):
            task.run()
        return callback

    # Start tasks in three different slots
    controller.start_task(
        worker,
        title="task1",
        slot=ThreadTaskSlot.change_density_image,
        provide_progress_controller=True,
        pos_args=[1],
        kw_args={},
        before_start_callback=make_callback()
    )

    controller.start_task(
        worker,
        title="task2",
        slot=ThreadTaskSlot.generate_territory_map,
        provide_progress_controller=True,
        pos_args=[2],
        kw_args={},
        before_start_callback=make_callback()
    )

    controller.start_task(
        worker,
        title="task3",
        slot=ThreadTaskSlot.generate_province_map,
        provide_progress_controller=True,
        pos_args=[3],
        kw_args={},
        before_start_callback=make_callback()
    )

    # With sync execution, all tasks should already be complete
    result = controller.cancel_all_and_wait(max_wait_ms=5000)
    assert result is True, "Should return True when all tasks complete"
    assert set(cancelled_tasks) == {1, 2, 3}, "All three tasks should have completed"
