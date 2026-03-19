# dsbf/utils/plot_factory.py

"""
PlotFactory module for DSBF

Centralizes generation of static and interactive plots using a standard schema,
consistent visual style, and dual rendering support (matplotlib + plotly).
"""

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, Union

import matplotlib
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.axes import Axes
from plotly.io import to_json

from dsbf.config.custom_plotly_templates import register_dark_theme
from dsbf.config.dashboard_theme_presets import THEME_CONFIGS, THEMES

register_dark_theme()

matplotlib.use("Agg")


class PlotData(TypedDict, total=False):
    """Standardized plot output schema used across DSBF."""

    type: Literal["histogram", "boxplot", "matrix", "line", "bar", "correlation"]
    data: dict[str, Any]
    config: dict[str, Any]
    annotations: list[str]


# --- Global style config ---
DEFAULT_PLOT_CONFIG = {
    "title_fontsize": 14,
    "label_fontsize": 12,
    "tick_labelsize": 10,
    "color": "#007acc",
    "figsize": (9, 5),
    "font": "DejaVu Sans",
    "tight_layout": True,
    "line_width": 2,
    "marker_size": 6,
}

STATIC_STYLE_RC = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#000000",
    "font.family": DEFAULT_PLOT_CONFIG["font"],
}


class PlotFactory:
    """Factory class for generating DSBF-compliant plots."""

    @staticmethod
    def _apply_static_style(
        ax: Axes,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        theme: str = "light",
    ) -> None:
        """
        Apply title, label, and tick styling to a Matplotlib axis.
        """
        cfg = THEME_CONFIGS[theme]

        ax.set_title(
            title,
            fontsize=DEFAULT_PLOT_CONFIG["title_fontsize"],
            weight="bold",
            color=cfg["fg_color"],
        )
        ax.set_xlabel(
            xlabel,
            fontsize=DEFAULT_PLOT_CONFIG["label_fontsize"],
            color=cfg["fg_color"],
        )
        ax.set_ylabel(
            ylabel,
            fontsize=DEFAULT_PLOT_CONFIG["label_fontsize"],
            color=cfg["fg_color"],
        )
        ax.tick_params(
            axis="both",
            labelsize=DEFAULT_PLOT_CONFIG["tick_labelsize"],
            colors=cfg["fg_color"],
        )

    @staticmethod
    def _apply_seaborn_style(theme: str = "light") -> None:
        """
        Apply DSBF-wide Seaborn + Matplotlib styling based on theme.
        """
        cfg = THEME_CONFIGS[theme]

        sns.set_theme(
            style="ticks",
            rc={
                "axes.facecolor": cfg["bg_color"],
                "figure.facecolor": cfg["bg_color"],
                "axes.edgecolor": cfg["fg_color"],
                "axes.labelcolor": cfg["fg_color"],
                "xtick.color": cfg["fg_color"],
                "ytick.color": cfg["fg_color"],
                "text.color": cfg["fg_color"],
                "grid.color": cfg["grid_color"],
                "font.family": DEFAULT_PLOT_CONFIG["font"],
            },
        )

        sns.set_context("notebook")
        sns.set_palette("deep")

    @staticmethod
    def _is_empty(data: Union[pd.Series, pd.DataFrame]) -> bool:
        return data.empty if isinstance(data, pd.DataFrame) else data.size == 0

    # === Truncate column name logic ===
    @staticmethod
    def _truncate(label, max_len) -> str:
        label = str(label)
        return label if len(label) <= max_len else label[: max_len - 3] + "..."

    @staticmethod
    def _truncate_unique(labels: list[str], max_len: int = 15) -> list[str]:
        # Gather truncated column names
        truncated = [PlotFactory._truncate(lbl, max_len=max_len) for lbl in labels]

        # Count each truncated column name
        counts = Counter(truncated)

        # Initialize disambiguated list
        seen = defaultdict(int)
        unique_labels = []

        # Loop over truncated column names
        for short in truncated:
            # If there are no duplicates
            if counts[short] == 1:
                unique_labels.append(short)

            # If there are duplicated truncated names
            else:
                seen[short] += 1
                unique_labels.append(f"{short} ({seen[short]})")

        return unique_labels

    @staticmethod
    def plot_histogram_static(
        series: pd.Series,
        save_path: str,
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if PlotFactory._is_empty(series):
            return {
                "path": {},
                "plot_data": {
                    "type": "histogram",
                    "data": {},
                    "config": {},
                    "annotations": ["Empty series"],
                },
            }

        x_label_str = str(series.name) if series.name else "Value"
        title_str = title or "Histogram"

        paths = {}

        for theme in THEMES:
            PlotFactory._apply_seaborn_style(theme)
            cfg = THEME_CONFIGS[theme]

            themed_path = Path(save_path).with_name(
                f"{Path(save_path).stem}_{theme}.png"
            )
            fig, ax = plt.subplots(figsize=DEFAULT_PLOT_CONFIG["figsize"])
            fig.patch.set_facecolor(cfg["bg_color"])
            ax.set_facecolor(cfg["bg_color"])

            df = pd.DataFrame({x_label_str: series})
            sns.histplot(
                data=df,
                x=x_label_str,
                bins=30,
                kde=False,
                color=cfg["accent_color"],
                ax=ax,
            )
            PlotFactory._apply_static_style(ax, title_str, x_label_str, "Count", theme)

            os.makedirs(themed_path.parent, exist_ok=True)
            fig.savefig(themed_path, bbox_inches="tight")
            plt.close(fig)

            paths[theme] = str(themed_path)

        return {
            "path": paths,
            "plot_data": {
                "type": "histogram",
                "data": {"x": series.tolist()},
                "config": {
                    "title": title_str,
                    "x_label": x_label_str,
                    "y_label": "Count",
                },
                "annotations": annotations or [],
            },
        }

    @staticmethod
    def plot_histogram_interactive(
        series: pd.Series,
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
        json_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        if PlotFactory._is_empty(series):
            return {
                "type": "histogram",
                "data": {},
                "config": {},
                "annotations": ["Empty series"],
            }

        x_label_str = str(series.name) if series.name else "Value"
        title_str = title or "Histogram"

        interactive_paths = {}

        for theme in THEMES:
            cfg = THEME_CONFIGS[theme]

            fig = go.Figure(
                [
                    go.Histogram(
                        x=series, nbinsx=30, marker=dict(color=cfg["accent_color"])
                    )
                ]
            )
            fig.update_layout(
                template=cfg["plotly_template"],
                title=dict(
                    text=title_str,
                    font=dict(
                        size=20,
                        family=DEFAULT_PLOT_CONFIG["font"],
                        color=cfg["fg_color"],
                    ),
                    x=0.5,
                    xanchor="center",
                ),
                xaxis=dict(
                    title=x_label_str,
                    title_font=dict(color=cfg["fg_color"]),
                    tickfont=dict(color=cfg["fg_color"]),
                    gridcolor=cfg["grid_color"],
                    showline=True,
                    linecolor=cfg["fg_color"],
                ),
                yaxis=dict(
                    title="Count",
                    title_font=dict(color=cfg["fg_color"]),
                    tickfont=dict(color=cfg["fg_color"]),
                    gridcolor=cfg["grid_color"],
                    showline=True,
                    linecolor=cfg["fg_color"],
                ),
                plot_bgcolor=cfg["bg_color"],
                paper_bgcolor=cfg["bg_color"],
                font=dict(color=cfg["fg_color"], family=DEFAULT_PLOT_CONFIG["font"]),
                margin=dict(l=40, r=20, t=60, b=40),
                height=360,
            )

            if json_path:
                themed_path = Path(json_path).with_name(
                    f"{Path(json_path).stem}_{theme}.json"
                )
                json_str = to_json(fig)
                if isinstance(json_str, str):
                    themed_path.write_text(json_str, encoding="utf-8")
                    interactive_paths[theme] = str(themed_path)
                else:
                    raise ValueError("Plotly to_json() returned non-str content.")

        return {
            "type": "histogram",
            "data": {"x": series.tolist()},
            "config": {
                "title": title_str,
                "x_label": x_label_str,
                "y_label": "Count",
            },
            "annotations": annotations or [],
            "interactive": interactive_paths if json_path else {},
        }

    @staticmethod
    def plot_boxplot_static(
        series: pd.Series,
        save_path: Union[str, Path],
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if PlotFactory._is_empty(series):
            return {
                "path": {},
                "plot_data": {
                    "type": "boxplot",
                    "data": {},
                    "config": {},
                    "annotations": ["Empty series"],
                },
            }

        x_label_str = str(series.name) if series.name else "Value"
        title_str = title or "Boxplot"
        paths = {}

        for theme in THEMES:
            PlotFactory._apply_seaborn_style(theme)
            cfg = THEME_CONFIGS[theme]

            themed_path = Path(save_path).with_name(
                f"{Path(save_path).stem}_{theme}.png"
            )
            fig, ax = plt.subplots(figsize=DEFAULT_PLOT_CONFIG["figsize"])
            fig.patch.set_facecolor(cfg["bg_color"])
            ax.set_facecolor(cfg["bg_color"])

            sns.boxplot(x=series, ax=ax, color=cfg["accent_color"])
            PlotFactory._apply_static_style(ax, title_str, x_label_str, "", theme)

            os.makedirs(themed_path.parent, exist_ok=True)
            fig.savefig(themed_path, bbox_inches="tight")
            plt.close(fig)

            paths[theme] = str(themed_path)

        return {
            "path": paths,
            "plot_data": {
                "type": "boxplot",
                "data": {"x": series.tolist()},
                "config": {"title": title_str, "x_label": x_label_str},
                "annotations": annotations or [],
            },
        }

    @staticmethod
    def plot_boxplot_interactive(
        series: pd.Series,
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
        json_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        if PlotFactory._is_empty(series):
            return {
                "type": "boxplot",
                "data": {},
                "config": {},
                "annotations": ["Empty series"],
            }

        x_label_str = str(series.name) if series.name else "Value"
        title_str = title or "Boxplot"
        interactive_paths = {}

        for theme in THEMES:
            cfg = THEME_CONFIGS[theme]

            fig = go.Figure(
                [
                    go.Box(
                        x=series,
                        name="",
                        marker_color=cfg["accent_color"],
                        boxpoints="outliers",
                    )
                ]
            )

            fig.update_layout(
                template=cfg["plotly_template"],
                title=dict(
                    text=title_str,
                    font=dict(
                        size=20,
                        family=DEFAULT_PLOT_CONFIG["font"],
                        color=cfg["fg_color"],
                    ),
                    x=0.5,
                    xanchor="center",
                ),
                xaxis=dict(
                    title=x_label_str,
                    title_font=dict(color=cfg["fg_color"]),
                    tickfont=dict(color=cfg["fg_color"]),
                    gridcolor=cfg["grid_color"],
                    showline=True,
                    linecolor=cfg["fg_color"],
                ),
                yaxis=dict(
                    title="",
                    title_font=dict(color=cfg["fg_color"]),
                    tickfont=dict(color=cfg["fg_color"]),
                    gridcolor=cfg["grid_color"],
                    showline=True,
                    linecolor=cfg["fg_color"],
                ),
                plot_bgcolor=cfg["bg_color"],
                paper_bgcolor=cfg["bg_color"],
                font=dict(color=cfg["fg_color"], family=DEFAULT_PLOT_CONFIG["font"]),
                margin=dict(l=40, r=20, t=60, b=40),
                height=360,
            )

            if json_path:
                themed_path = Path(json_path).with_name(
                    f"{Path(json_path).stem}_{theme}.json"
                )
                json_str = to_json(fig)
                if isinstance(json_str, str):
                    themed_path.write_text(json_str, encoding="utf-8")
                    interactive_paths[theme] = str(themed_path)
                else:
                    raise ValueError("Plotly to_json() returned non-str content.")

        return {
            "type": "boxplot",
            "data": {"x": series.tolist()},
            "config": {
                "title": title_str,
                "x_label": x_label_str,
            },
            "annotations": annotations or [],
            "interactive": interactive_paths if json_path else {},
        }

    @staticmethod
    def plot_barplot_static(
        series: pd.Series,
        save_path: Union[str, Path],
        top_k: Optional[int] = 100,
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if PlotFactory._is_empty(series):
            return {
                "path": {},
                "plot_data": {
                    "type": "barplot",
                    "data": {},
                    "config": {},
                    "annotations": ["Empty series"],
                },
            }

        counts = series.value_counts(dropna=False)
        if top_k is not None:
            counts = counts.nlargest(top_k)
        x_vals = counts.index.astype(str)
        y_vals = counts.values

        title_str = title or "Top Categories"
        x_label_str = str(series.name) if series.name else "Category"

        paths = {}

        for theme in THEMES:
            PlotFactory._apply_seaborn_style(theme)
            cfg = THEME_CONFIGS[theme]

            themed_path = Path(save_path).with_name(
                f"{Path(save_path).stem}_{theme}.png"
            )
            fig, ax = plt.subplots(figsize=DEFAULT_PLOT_CONFIG["figsize"])
            fig.patch.set_facecolor(cfg["bg_color"])
            ax.set_facecolor(cfg["bg_color"])

            sns.barplot(x=x_vals, y=y_vals, ax=ax, color=cfg["accent_color"])

            PlotFactory._apply_static_style(
                ax,
                title=title_str,
                xlabel=x_label_str,
                ylabel="Count",
                theme=theme,
            )

            ax.tick_params(axis="x", rotation=45)

            os.makedirs(themed_path.parent, exist_ok=True)
            fig.savefig(themed_path, bbox_inches="tight")
            plt.close(fig)

            paths[theme] = str(themed_path)

        return {
            "path": paths,
            "plot_data": {
                "type": "barplot",
                "data": {"x": x_vals.tolist(), "y": y_vals.tolist()},
                "config": {
                    "title": title_str,
                    "x_label": x_label_str,
                    "y_label": "Count",
                },
                "annotations": annotations or [],
            },
        }

    @staticmethod
    def plot_barplot_interactive(
        series: pd.Series,
        top_k: Optional[int] = 100,
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
        json_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        if PlotFactory._is_empty(series):
            return {
                "type": "barplot",
                "data": {},
                "config": {},
                "annotations": ["Empty series"],
            }

        counts = series.value_counts(dropna=False)
        if top_k is not None:
            counts = counts.nlargest(top_k)

        raw_labels = counts.index.astype(str).tolist()
        x_vals = PlotFactory._truncate_unique(raw_labels, max_len=15)
        y_vals = counts.values

        title_str = title or "Top Categories"
        x_label_str = str(series.name) if series.name else "Category"

        interactive_paths = {}

        for theme in THEMES:
            cfg = THEME_CONFIGS[theme]

            fig = go.Figure(
                [
                    go.Bar(
                        x=x_vals,
                        y=y_vals,
                        marker_color=cfg["accent_color"],
                        hovertext=raw_labels,
                        hoverinfo="text+y",
                    )
                ]
            )

            fig.update_layout(
                template=cfg["plotly_template"],
                title=title_str,
                xaxis_title=x_label_str,
            )
            fig.update_xaxes(autotickangles=[45, 60, 90])

            if json_path:
                themed_path = Path(json_path).with_name(
                    f"{Path(json_path).stem}_{theme}.json"
                )
                json_str = to_json(fig)
                if isinstance(json_str, str):
                    themed_path.write_text(json_str, encoding="utf-8")
                    interactive_paths[theme] = str(themed_path)
                else:
                    raise ValueError("Plotly to_json() returned non-str content.")

        return {
            "type": "barplot",
            "data": {"x": x_vals, "y": y_vals.tolist()},
            "config": {
                "title": title_str,
                "x_label": x_label_str,
                "y_label": "Count",
            },
            "annotations": annotations or [],
            "interactive": interactive_paths if json_path else {},
        }

    @staticmethod
    def plot_null_matrix_static(
        df: pd.DataFrame,
        save_path: str,
        title: Optional[str] = "Null Matrix",
        annotations: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        PlotFactory._apply_seaborn_style()

        if PlotFactory._is_empty(df):
            return {
                "path": Path(save_path),
                "plot_data": {
                    "type": "matrix",
                    "data": {},
                    "config": {},
                    "annotations": ["Empty dataframe"],
                },
            }

        title_str = title or "Null Matrix"
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        ax.set_title(title_str)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

        return {
            "path": Path(save_path),
            "plot_data": {
                "type": "matrix",
                "data": {},
                "config": {"title": title_str},
                "annotations": annotations or [],
            },
        }

    @staticmethod
    def plot_null_matrix_interactive(
        df: pd.DataFrame,
        title: Optional[str] = "Null Matrix",
        annotations: Optional[list[str]] = None,
        json_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        if PlotFactory._is_empty(df):
            return {
                "type": "matrix",
                "data": {},
                "config": {},
                "annotations": ["Empty dataframe"],
            }

        title_str = title or "Null Matrix"
        z = df.isnull().astype(int).values
        fig = go.Figure([go.Heatmap(z=z, colorscale="Viridis")])
        fig.update_layout(title=title_str)

        if json_path:
            json_path = Path(json_path)
            json_str = to_json(fig)
            if isinstance(json_str, str):
                json_path.write_text(json_str, encoding="utf-8")
            else:
                raise ValueError("Plotly to_json() returned non-str content.")

        return {
            "type": "matrix",
            "data": {},
            "config": {"title": title_str},
            "annotations": annotations or [],
        }

    @staticmethod
    def plot_correlation_static(
        df: pd.DataFrame,
        save_path: Union[str, Path],
        title: Optional[str] = "Correlation Matrix",
        annotations: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if df.empty:
            return {
                "path": {},
                "plot_data": {
                    "type": "correlation_matrix",
                    "data": {},
                    "config": {},
                    "annotations": ["Empty DataFrame"],
                },
            }

        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        paths = {}

        for theme in THEMES:
            PlotFactory._apply_seaborn_style(theme)
            cfg = THEME_CONFIGS[theme]

            themed_path = Path(save_path).with_name(
                f"{Path(save_path).stem}_{theme}.png"
            )
            fig, ax = plt.subplots(figsize=DEFAULT_PLOT_CONFIG["figsize"])
            fig.patch.set_facecolor(cfg["bg_color"])
            ax.set_facecolor(cfg["bg_color"])

            sns.heatmap(
                corr,
                mask=mask,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".2f",
                square=True,
                linewidths=0.5,
                linecolor=cfg["grid_color"],
                cbar_kws={"shrink": 0.75},
                ax=ax,
            )

            ax.set_title(
                title or "Correlation Matrix",
                fontsize=DEFAULT_PLOT_CONFIG["title_fontsize"],
                color=cfg["fg_color"],
            )
            ax.tick_params(
                axis="both",
                labelsize=DEFAULT_PLOT_CONFIG["tick_labelsize"],
                colors=cfg["fg_color"],
            )

            os.makedirs(themed_path.parent, exist_ok=True)
            fig.savefig(themed_path, bbox_inches="tight")
            plt.close(fig)

            paths[theme] = str(themed_path)

        return {
            "path": paths,
            "plot_data": {
                "type": "correlation_matrix",
                "data": corr.to_dict(),
                "config": {"title": title},
                "annotations": annotations or [],
            },
        }

    @staticmethod
    def plot_correlation_interactive(
        df: pd.DataFrame,
        title: Optional[str] = "Correlation Matrix",
        annotations: Optional[list[str]] = None,
        json_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        corr = df.corr(numeric_only=True)
        if corr.empty:
            return {
                "type": "correlation",
                "data": {},
                "config": {},
                "annotations": ["No numeric columns"],
            }

        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.mask(mask)

        # Truncate axis labels
        diag_labels = corr.columns.tolist()
        axis_labels_display = PlotFactory._truncate_unique(diag_labels)

        # Clear diagonal values
        for i in range(len(diag_labels)):
            corr_masked.iat[i, i] = None

        # Hover text matrix
        hover_text = []
        for i, row in enumerate(corr_masked.values):
            hover_row = []
            for j, val in enumerate(row):
                hover_row.append("" if pd.isnull(val) else f"{val:.2f}")
            hover_text.append(hover_row)

        # Adaptive font sizing
        annot_font_size = max(6, min(18, int(150 / len(corr.columns))))
        interactive_paths = {}

        for theme in THEMES:
            cfg = THEME_CONFIGS[theme]

            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_masked.values,
                    x=axis_labels_display,
                    y=axis_labels_display,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                    text=hover_text,
                    texttemplate="%{text}",
                    textfont={"size": annot_font_size},
                    hoverinfo="text",
                    colorbar=dict(title="Correlation"),
                )
            )

            fig.update_layout(
                template=cfg["plotly_template"],
                title=dict(
                    text=title or "Correlation Matrix",
                    font=dict(
                        size=20,
                        family=DEFAULT_PLOT_CONFIG["font"],
                        color=cfg["fg_color"],
                    ),
                    x=0.5,
                    xanchor="center",
                ),
                xaxis=dict(
                    tickangle=45,
                    side="bottom",
                    tickfont=dict(size=annot_font_size, color=cfg["fg_color"]),
                    showline=True,
                    linecolor=cfg["fg_color"],
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(size=annot_font_size, color=cfg["fg_color"]),
                    showline=True,
                    linecolor=cfg["fg_color"],
                ),
                plot_bgcolor=cfg["bg_color"],
                paper_bgcolor=cfg["bg_color"],
                font=dict(color=cfg["fg_color"], family=DEFAULT_PLOT_CONFIG["font"]),
                autosize=True,
                margin=dict(l=80, r=80, t=80, b=80, autoexpand=True),
            )

            if json_path:
                themed_path = Path(json_path).with_name(
                    f"{Path(json_path).stem}_{theme}.json"
                )
                json_str = to_json(fig)
                if isinstance(json_str, str):
                    themed_path.write_text(json_str, encoding="utf-8")
                    interactive_paths[theme] = str(themed_path)
                else:
                    raise ValueError("Plotly to_json() returned non-str content.")

        return {
            "type": "correlation",
            "data": {
                str(k): {str(inner_k): v for inner_k, v in inner_dict.items()}
                for k, inner_dict in corr_masked.to_dict().items()
            },
            "config": {"title": title or "Correlation Matrix"},
            "annotations": annotations or [],
            "interactive": interactive_paths if json_path else {},
        }

    @staticmethod
    def plot_missingness_matrix(
        df: pd.DataFrame,
        save_path: str,
        title: Optional[str] = "Missingness Matrix",
        annotations: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Uses missingno to generate a missingness matrix plot. Returns static path only.

        Args:
            df (pd.DataFrame): DataFrame to visualize.
            save_path (str): Path to save static image.
            title (Optional[str]): Optional title (not directly used by missingno).
            annotations (Optional[list[str]]):

        Returns:
            dict: dict with "path" key pointing to saved image.
        """
        if PlotFactory._is_empty(df):
            return {
                "type": "matrix",
                "data": {},
                "config": {},
                "annotations": ["Empty dataframe"],
            }

        title_str = title or "Missingness Matrix"
        plt.figure()
        msno.matrix(df, labels=True)
        # plt.title(title_str)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        return {
            "path": Path(save_path),
            "plot_data": {
                "type": "matrix",
                "data": {},
                "config": {"title": title_str},
                "annotations": annotations or [],
            },
        }

    @staticmethod
    def plot_stacked_bar_interactive(
        df: pd.DataFrame,
        title: str = "Dtype Mapping: Inferred vs Intent",
        annotations: Optional[list[str]] = None,
        json_path: Optional[Union[str, Path]] = None,
        color_map: Optional[dict[str, str]] = {},
    ) -> dict[str, Any]:
        if df.empty or not {"analysis_intent_dtype", "inferred_dtype"}.issubset(
            df.columns
        ):
            return {
                "type": "bar",
                "data": {},
                "config": {},
                "annotations": ["Missing required dtype columns"],
            }

        # Grouped counts
        counts = (
            df.groupby(["analysis_intent_dtype", "inferred_dtype"])
            .size()
            .reset_index(name="count")
        )

        # Pivoted format
        pivot = counts.pivot_table(
            index="analysis_intent_dtype",
            columns="inferred_dtype",
            values="count",
            fill_value=0,
        )

        color_map = color_map or {}
        interactive_paths = {}

        for theme in THEMES:
            cfg = THEME_CONFIGS[theme]

            # Bar traces
            fig = go.Figure(
                [
                    go.Bar(
                        name=str(col),
                        x=pivot.index.tolist(),
                        y=pivot[col].tolist(),
                        text=pivot[col].tolist(),
                        textposition="inside",
                        insidetextfont={"size": 16, "color": "black"},
                        marker_color=color_map.get(str(col), cfg["accent_color"]),
                    )
                    for col in pivot.columns
                ]
            )

            # Totals on top
            totals = pivot.sum(axis=1)
            for x, total in zip(pivot.index.tolist(), totals.tolist()):
                fig.add_annotation(
                    x=x,
                    y=total + 1,
                    text=f"Total: {total}",
                    showarrow=False,
                    font=dict(size=14, color=cfg["fg_color"]),
                    yanchor="bottom",
                )

            fig.update_layout(
                barmode="stack",
                plot_bgcolor=cfg["bg_color"],
                paper_bgcolor=cfg["bg_color"],
                title=dict(
                    text=title,
                    font=dict(
                        size=20,
                        family=DEFAULT_PLOT_CONFIG["font"],
                        color=cfg["fg_color"],
                    ),
                    x=0.5,
                    xanchor="center",
                ),
                xaxis=dict(
                    title="Analysis Intent Dtype",
                    title_font=dict(size=18, color=cfg["fg_color"]),
                    tickfont=dict(size=14, color=cfg["fg_color"]),
                    showline=True,
                    linecolor=cfg["fg_color"],
                ),
                yaxis=dict(
                    title="Count",
                    title_font=dict(size=18, color=cfg["fg_color"]),
                    tickfont=dict(size=14, color=cfg["fg_color"]),
                    showline=True,
                    linecolor=cfg["fg_color"],
                    gridcolor=cfg["grid_color"],
                ),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=cfg["fg_color"]),
                ),
                font=dict(family=DEFAULT_PLOT_CONFIG["font"], color=cfg["fg_color"]),
                margin=dict(l=40, r=20, t=60, b=40),
                height=360,
            )

            if json_path:
                themed_path = Path(json_path).with_name(
                    f"{Path(json_path).stem}_{theme}.json"
                )
                json_str = to_json(fig)
                if isinstance(json_str, str):
                    themed_path.write_text(json_str, encoding="utf-8")
                    interactive_paths[theme] = str(themed_path)
                else:
                    raise ValueError("Plotly to_json() returned non-str content.")

        return {
            "type": "bar",
            "data": {str(k): v for k, v in counts.to_dict(orient="list").items()},
            "config": {
                "title": title,
                "x_label": "analysis_intent_dtype",
                "y_label": "count",
            },
            "annotations": annotations or [],
            "interactive": interactive_paths if json_path else {},
        }

    @staticmethod
    def plot_boxplot_hist_composite_static(
        series: pd.Series,
        save_path: Union[str, Path],
        kde: bool = True,
        bins: Optional[int] = None,
        annotations: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Create two composite static plots per theme:
        1. Boxplot above histogram
        2. Histogram above boxplot
        Returns paths grouped by theme.
        """
        if PlotFactory._is_empty(series):
            return {}

        x = series.dropna().to_numpy()
        x_label = str(series.name) if series.name else "Value"
        mean_val = float(np.mean(x))
        theme_to_paths = {}

        for theme in THEMES:
            PlotFactory._apply_seaborn_style(theme)
            cfg = THEME_CONFIGS[theme]
            paths = {}

            for layout in ["box_above", "hist_above"]:
                fig = plt.figure(
                    figsize=DEFAULT_PLOT_CONFIG["figsize"], constrained_layout=True
                )
                gs = fig.add_gridspec(2, 1, height_ratios=[0.2, 0.8], hspace=0.02)

                if layout == "box_above":
                    ax_box = fig.add_subplot(gs[0], xticks=[])
                    ax_hist = fig.add_subplot(gs[1], ylabel="Density", xlabel=x_label)
                else:
                    ax_hist = fig.add_subplot(gs[0], yticks=[], ylabel=None, xticks=[])
                    ax_box = fig.add_subplot(gs[1], xlabel=x_label)

                fig.patch.set_facecolor(cfg["bg_color"])
                ax_box.set_facecolor(cfg["bg_color"])
                ax_hist.set_facecolor(cfg["bg_color"])

                sns.boxplot(x=x, ax=ax_box, color=cfg["accent_color"])
                ax_box.set(yticks=[], ylabel=None)

                sns.histplot(
                    x=x,
                    kde=kde,
                    stat="density",
                    bins=bins or "auto",
                    ax=ax_hist,
                    color=cfg["accent_color"],
                )

                annotation_text = f"Mean: {mean_val:.2f}"
                if layout == "box_above":
                    ax_hist.axvline(
                        mean_val,
                        color="darkred",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.8,
                    )
                    ax_hist.annotate(
                        annotation_text,
                        xy=(mean_val, ax_hist.get_ylim()[1]),
                        xytext=(6, -10),
                        textcoords="offset points",
                        ha="left",
                        va="top",
                        fontsize=14,
                        color="darkred",
                        fontweight="bold",
                    )
                else:
                    ax_box.axvline(
                        mean_val,
                        color="darkred",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.8,
                    )
                    ax_box.annotate(
                        annotation_text,
                        xy=(mean_val, ax_box.get_ylim()[1]),
                        xytext=(6, -10),
                        textcoords="offset points",
                        ha="left",
                        va="top",
                        fontsize=14,
                        color="darkred",
                        fontweight="bold",
                    )

                if layout == "box_above":
                    sns.despine(ax=ax_box, left=True, bottom=True)
                    sns.despine(ax=ax_hist)
                else:
                    sns.despine(ax=ax_box, left=True)
                    sns.despine(ax=ax_hist, left=True, bottom=True)

                plot_path = Path(save_path).with_name(
                    f"{Path(save_path).stem}_{layout}_{theme}.png"
                )
                os.makedirs(plot_path.parent, exist_ok=True)
                fig.savefig(str(plot_path), bbox_inches="tight")
                plt.close(fig)

                paths[layout] = str(plot_path)

            theme_to_paths[theme] = paths

        return theme_to_paths
