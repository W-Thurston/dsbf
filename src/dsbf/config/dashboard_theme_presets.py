# dsbf/config/dashboard_theme_presets.py

"""
Theme presets for dashboard and plot rendering.

Each theme defines background, foreground, accent, and grid colors,
as well as Plotly and Matplotlib-specific settings.

These themes are used by:
- PlotFactory (static and interactive chart rendering)
- Panel dashboard styling
- Future support for user-configurable themes
"""

THEMES: list[str] = ["light", "dark"]

THEME_CONFIGS: dict[str, dict[str, str]] = {
    "light": {
        "bg_color": "#ffffff",
        "fg_color": "#000000",
        "accent_color": "#007acc",
        "grid_color": "#e0e0e0",
        "plotly_template": "plotly_white",
        "annotation_color": "#000000",
        "mean_line_color": "darkred",
    },
    "dark": {
        "bg_color": "#292929",  # Dashboard card color
        "fg_color": "#e2e2e2",  # Softer white for less contrast
        "accent_color": "#1f77b4",  # Same blue as light mode
        "grid_color": "#444444",  # Soft dark grid
        "plotly_template": "simple_dark",  # We manually match simple_white structure
        "annotation_color": "#e2e2e2",  # Match fg
        "mean_line_color": "#ff6b6b",  # Contrast but still visible
        "tick_style": "outside",  # Matches simple_white
        "line_color": "#e2e2e2",  # Axis and shape lines
        "bar_outline_color": "#444444",  # Hist/bar outlines like white in simple_white
        "font_family": "Arial",  # Match simple_white
        "title_align": "center",  # Keep centered unless you want `left`
    },
}
