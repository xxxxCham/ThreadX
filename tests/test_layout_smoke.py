"""
ThreadX UI Tests - Layout Smoke Tests
======================================

Tests de fumée pour vérifier que le layout principal se charge
correctement et contient les éléments de base attendus.

Vérifie:
    - App démarre sans erreur
    - Layout non-None
    - 4 onglets présents (Data, Indicators, Backtest, Optimization)
    - Container racine Bootstrap

Author: ThreadX Framework
Version: Prompt 8 - Tests & Qualité
"""

import pytest

pytestmark = pytest.mark.ui


def test_app_layout_exists(dash_app):
    """Test that Dash app has a valid layout."""
    assert dash_app is not None, "Dash app should be created"
    assert dash_app.layout is not None, "App layout should not be None"


def test_main_container_exists(dash_app):
    """Test that main container (Bootstrap) exists in layout."""
    import dash_bootstrap_components as dbc

    layout = dash_app.layout

    # Layout root should be dbc.Container
    assert isinstance(layout, dbc.Container), "Root layout should be dbc.Container"

    # Should have fluid=True and dark theme classes
    assert layout.fluid is True, "Container should be fluid"
    assert "bg-dark" in layout.className, "Container should have bg-dark class"


def test_tabs_present(dash_app):
    """Test that all 4 main tabs are present in layout."""
    from dash import dcc

    from tests.conftest import find_component_by_id

    layout = dash_app.layout

    # Find main tabs component
    tabs = find_component_by_id(layout, "main-tabs")
    assert tabs is not None, "main-tabs component should exist"
    assert isinstance(tabs, dcc.Tabs), "main-tabs should be dcc.Tabs"

    # Verify 4 tabs
    assert hasattr(tabs, "children"), "Tabs should have children"
    tab_children = tabs.children
    assert isinstance(tab_children, list), "Tabs children should be list"
    assert len(tab_children) == 4, f"Should have 4 tabs, found {len(tab_children)}"

    # Verify tab values (IDs)
    expected_values = [
        "tab-data",
        "tab-indicators",
        "tab-backtest",
        "tab-optimization",
    ]
    actual_values = [tab.value for tab in tab_children]
    assert actual_values == expected_values, f"Tab values mismatch: {actual_values}"


def test_stores_present(dash_app):
    """Test that task stores are present for async polling."""
    from dash import dcc

    from tests.conftest import find_component_by_id

    layout = dash_app.layout

    # Check all 4 task stores
    stores = [
        "data-task-store",
        "indicators-task-store",
        "bt-task-store",
        "opt-task-store",
    ]

    for store_id in stores:
        store = find_component_by_id(layout, store_id)
        assert store is not None, f"Store '{store_id}' should exist in layout"
        assert isinstance(store, dcc.Store), f"'{store_id}' should be dcc.Store"


def test_interval_present(dash_app):
    """Test that global polling interval exists."""
    from dash import dcc

    from tests.conftest import find_component_by_id

    layout = dash_app.layout

    # Check global interval
    interval = find_component_by_id(layout, "global-interval")
    assert interval is not None, "global-interval should exist"
    assert isinstance(interval, dcc.Interval), "Should be dcc.Interval"
    assert interval.interval == 500, "Interval should be 500ms"


def test_header_present(dash_app):
    """Test that header (Navbar) is present."""
    import dash_bootstrap_components as dbc

    layout = dash_app.layout

    # Find navbar (first dbc.Navbar in layout)
    def find_navbar(component):
        if isinstance(component, dbc.Navbar):
            return component
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                result = find_navbar(child)
                if result:
                    return result
        return None

    navbar = find_navbar(layout)
    assert navbar is not None, "Navbar should exist in header"
    assert navbar.color == "dark", "Navbar should have dark theme"
    assert navbar.dark is True, "Navbar should be dark variant"


def test_footer_present(dash_app):
    """Test that footer is present."""
    from dash import html

    layout = dash_app.layout

    # Find footer
    def find_footer(component):
        if isinstance(component, html.Footer):
            return component
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                result = find_footer(child)
                if result:
                    return result
        return None

    footer = find_footer(layout)
    assert footer is not None, "Footer should exist"
