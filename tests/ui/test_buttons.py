import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from opengs_maptool.ui.buttons import TimeoutSlider


def test_timeout_slider_restarts_cooldown_on_each_change():
    app = QApplication.instance() or QApplication([])
    calls = []
    slider = TimeoutSlider(
        Qt.Orientation.Horizontal,
        minimum=0,
        maximum=10,
        default=0,
        tick_interval=1,
        step=1,
        onchange=calls.append,
        cooldown_ms=100,
    )

    slider.setValue(1)
    QTest.qWait(70)
    slider.setValue(2)
    QTest.qWait(70)
    assert calls == []

    QTest.qWait(50)
    assert calls == [2]
