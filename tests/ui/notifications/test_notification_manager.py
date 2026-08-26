from unittest.mock import MagicMock

from opengs_maptool.ui.notifications import notification_manager
from opengs_maptool.models.progress_status import ProgressStatus


class FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def emit(self, *args, **kwargs):
        for cb in list(self._callbacks):
            cb(*args, **kwargs)


class FakeTaskSignals:
    def __init__(self):
        self.task_progress_started = FakeSignal()
        self.task_progress_phase_started = FakeSignal()
        self.task_progress_updated = FakeSignal()
        self.task_progress_retired = FakeSignal()
        self.task_cancelled = FakeSignal()
        self.task_cancel_requested = FakeSignal()
        self.task_successful = FakeSignal()
        self.task_error = FakeSignal()


class FakeTask:
    def __init__(self, title="t1"):
        self.title = title
        self.signals = FakeTaskSignals()


class FakeLayout:
    def __init__(self):
        self.added = []

    def addWidget(self, widget):
        self.added.append(widget)


def make_toast_mock():
    m = MagicMock()
    # Provide the methods NotificationManager expects
    for name in (
        'set_task_signals', 'on_progress_started', 'on_phase_started',
        'on_progress_updated', 'on_retired', 'on_cancelled', 'on_cancel_requested',
        'on_success', 'on_error',
    ):
        setattr(m, name, MagicMock())
    return m


def test_new_task_wires_signals_and_defers_show(monkeypatch):
    # Prepare fake controller with a new_task_started signal
    fake_controller = MagicMock()
    fake_controller.new_task_started = FakeSignal()

    # Replace TaskToastWidget to return a mock
    toast_mock = make_toast_mock()
    monkeypatch.setattr(notification_manager, 'TaskToastWidget', lambda title, main_window: toast_mock)

    layout = FakeLayout()
    nm = notification_manager.NotificationManager(main_window=None, task_controller=fake_controller, toast_container_layout=layout)

    task = FakeTask(title="My Task")

    # Emit new task started
    fake_controller.new_task_started.emit(task)

    # A toast should have been created and stored in the pending list
    assert any(t is task for (t, w) in nm._tasks_without_progress)

    # set_task_signals should have been called so the toast can send cancel requests
    toast_mock.set_task_signals.assert_called_once()

    # The task signals should have listeners connected (check that the toast handler is present)
    callbacks = task.signals.task_progress_started._callbacks
    assert any(cb == toast_mock.on_progress_started for cb in callbacks)


def test_first_progress_start_adds_widget_and_removes_pending(monkeypatch):
    fake_controller = MagicMock()
    fake_controller.new_task_started = FakeSignal()

    toast_mock = make_toast_mock()
    monkeypatch.setattr(notification_manager, 'TaskToastWidget', lambda title, main_window: toast_mock)

    layout = FakeLayout()
    nm = notification_manager.NotificationManager(main_window=None, task_controller=fake_controller, toast_container_layout=layout)

    task = FakeTask(title="T2")
    fake_controller.new_task_started.emit(task)

    # simulate the first progress start
    status = ProgressStatus()
    task.signals.task_progress_started.emit(status)

    # The toast should have been added to the layout and received the status
    assert layout.added and layout.added[0] is toast_mock
    toast_mock.on_progress_started.assert_called_with(status)

    # The pending list should no longer contain the task
    assert all(t is not task for (t, w) in nm._tasks_without_progress)



def test_retire_without_cancel_request_does_not_show_cancelling(monkeypatch):
    """A ProgressController retires on normal completion too, not only on cancel."""
    from opengs_maptool.ui.notifications.task_toast_widget import TaskToastWidget

    toast = TaskToastWidget.__new__(TaskToastWidget)  # skip Qt widget construction
    toast._is_active = True
    toast._cancel_requested = False
    toast.status_label = MagicMock()
    toast.close_btn = MagicMock()
    toast._set_status_tone = MagicMock()

    toast.on_retired()

    toast.status_label.setText.assert_not_called()
    toast.close_btn.setEnabled.assert_not_called()


def test_retire_after_cancel_request_shows_cancelling():
    from opengs_maptool.ui.notifications.task_toast_widget import TaskToastWidget

    toast = TaskToastWidget.__new__(TaskToastWidget)
    toast._is_active = True
    toast._cancel_requested = True
    toast.status_label = MagicMock()
    toast.close_btn = MagicMock()
    toast._set_status_tone = MagicMock()

    toast.on_retired()

    toast.status_label.setText.assert_called_once_with("Cancelling...")
    toast.close_btn.setEnabled.assert_called_once_with(False)
