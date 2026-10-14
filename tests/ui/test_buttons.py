import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QHBoxLayout, QSlider, QWidget

from opengs_maptool.ui.buttons import create_slider


def test_slider_calls_callback_immediately_when_value_changes():
    app = QApplication.instance() or QApplication([])
    calls = []
    parent = QWidget()
    layout = QHBoxLayout(parent)
    slider = create_slider(
        layout,
        label_text="Value:",
        minimum=0,
        maximum=10,
        default=0,
        tick_interval=1,
        step=1,
        onchange_callback_function=calls.append,
    )

    slider.setValue(1)
    assert isinstance(slider, QSlider)
    assert calls == [1]
