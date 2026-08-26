import os

# Must be set before any QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QGroupBox, QPushButton

from opengs_maptool.app import App
from opengs_maptool.simple_types import TabName


@pytest.fixture(scope="module")
def app():
    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    application = App()
    yield application


@pytest.fixture
def window(app):
    from opengs_maptool.ui.main_window import MainWindow
    return MainWindow(app)


def test_left_panel_is_populated_for_the_initial_tab(window):
    """addTab() selects the first tab before currentChanged is connected.

    Nothing repopulates the left panel afterwards, so without an explicit
    initial refresh the panel stays blank until the user switches tabs.
    """
    left_panel = window._left_panel

    assert window._tabs.currentIndex() == 0
    assert left_panel._current_tab_name is TabName.LAND
    assert left_panel._content_layout.count() > 0, "left panel is blank on startup"

    button_labels = [b.text() for b in left_panel.findChildren(QPushButton)]
    assert "Import Land Image" in button_labels


def test_switching_tabs_replaces_left_panel_content(window):
    left_panel = window._left_panel

    window._tabs.setCurrentIndex(window._tab_names_by_index.index(TabName.DENSITY))
    assert left_panel._current_tab_name is TabName.DENSITY
    labels = [b.text() for b in left_panel.findChildren(QPushButton)]
    assert "Equator Distribution" in labels

    window._tabs.setCurrentIndex(window._tab_names_by_index.index(TabName.LAND))
    assert left_panel._current_tab_name is TabName.LAND
    assert left_panel._content_layout.count() > 0
    assert any(g.title() == "Informations" for g in left_panel.findChildren(QGroupBox))


class _StuckThreadPool:
    """Starts nothing, so the slot stays occupied for the duration of a test."""
    def start(self, runnable):
        return


@pytest.fixture
def occupied_generation_slot(monkeypatch):
    """Hold a generation slot open without running anything."""
    from opengs_maptool.controllers import task_controller as tc_module

    monkeypatch.setattr(
        tc_module.QThreadPool, 'globalInstance', staticmethod(lambda: _StuckThreadPool())
    )

    occupied = []

    def occupy(window, slot):
        occupied.append((window._context.task_controller, slot))
        return window._context.task_controller.start_task(
            lambda progress_controller: None,
            title="busy", slot=slot, pos_args=[], kw_args={},
        )

    yield occupy

    # The TaskController lives on the shared application context, so a slot left
    # occupied would leak into the next test.
    for controller, slot in occupied:
        controller._free_slot(slot)


def _labels(left_panel):
    return {b.text(): b.isEnabled() for b in left_panel.findChildren(QPushButton)
            if not b.isHidden() or True}


def test_guardrails_disable_inputs_while_a_map_generates(window, occupied_generation_slot, monkeypatch):
    from opengs_maptool.controllers.task_controller import ThreadTaskSlot
    import opengs_maptool.config as config

    monkeypatch.setattr(config, "GUARDRAILS", True)
    left_panel = window._left_panel

    window._tabs.setCurrentIndex(window._tab_names_by_index.index(TabName.LAND))
    assert _labels(left_panel)["Import Land Image"] is True

    occupied_generation_slot(window, ThreadTaskSlot.generate_territory_map)

    # The tab on screen is locked...
    assert _labels(left_panel)["Import Land Image"] is False
    # ...and so is a tab visited while the job runs.
    window._tabs.setCurrentIndex(window._tab_names_by_index.index(TabName.DENSITY))
    density = _labels(left_panel)
    assert density["Import Density Image"] is False
    assert density["Normalize Density"] is False


def test_generating_a_territory_map_blocks_generating_provinces(window, occupied_generation_slot, monkeypatch):
    from opengs_maptool.controllers.task_controller import ThreadTaskSlot
    import opengs_maptool.config as config

    monkeypatch.setattr(config, "GUARDRAILS", True)
    occupied_generation_slot(window, ThreadTaskSlot.generate_territory_map)

    window._tabs.setCurrentIndex(window._tab_names_by_index.index(TabName.PROVINCE))
    assert _labels(window._left_panel)["Generate Provinces"] is False


def test_guardrails_disabled_leaves_inputs_alone(window, occupied_generation_slot, monkeypatch):
    from opengs_maptool.controllers.task_controller import ThreadTaskSlot
    import opengs_maptool.config as config

    monkeypatch.setattr(config, "GUARDRAILS", False)
    left_panel = window._left_panel

    occupied_generation_slot(window, ThreadTaskSlot.generate_territory_map)

    window._tabs.setCurrentIndex(window._tab_names_by_index.index(TabName.LAND))
    assert _labels(left_panel)["Import Land Image"] is True


def test_guardrails_release_when_the_slot_frees(window, occupied_generation_slot, monkeypatch):
    from opengs_maptool.controllers.task_controller import ThreadTaskSlot
    import opengs_maptool.config as config

    monkeypatch.setattr(config, "GUARDRAILS", True)
    left_panel = window._left_panel
    controller = window._context.task_controller

    window._tabs.setCurrentIndex(window._tab_names_by_index.index(TabName.LAND))
    occupied_generation_slot(window, ThreadTaskSlot.generate_territory_map)
    assert _labels(left_panel)["Import Land Image"] is False

    # However the task ends -- success, error or cancel -- the slot is freed.
    controller._free_slot(ThreadTaskSlot.generate_territory_map)
    assert _labels(left_panel)["Import Land Image"] is True
