# dsbf/utils/plot_factory.py

"""
PlotFactory module for DSBF

Centralizes generation of static and interactive plots using a standard schema,
consistent visual style, and dual rendering support (matplotlib + plotly).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import matplotlib
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.axes import Axes

matplotlib.use("Agg")


class PlotData(TypedDict, total=False):
    """Standardized plot output schema used across DSBF."""

    type: Literal["histogram", "boxplot", "matrix", "line", "bar", "correlation"]
    data: Dict[str, Any]
    config: Dict[str, Any]
    annotations: List[str]


# --- Global style config ---
DEFAULT_PLOT_CONFIG = {
    "title_fontsize": 14,
    "label_fontsize": 12,
    "tick_labelsize": 10,
    "color": "#007acc",
    "figsize": (6, 4),
    "font": "DejaVu Sans",
    "tight_layout": True,
}


def apply_static_style(
    ax: Axes, title: str = "", xlabel: str = "", ylabel: str = ""
) -> None:
    """Apply DSBF-standard style to a static matplotlib plot."""
    ax.set_title(title, fontsize=DEFAULT_PLOT_CONFIG["title_fontsize"])
    ax.set_xlabel(xlabel, fontsize=DEFAULT_PLOT_CONFIG["label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=DEFAULT_PLOT_CONFIG["label_fontsize"])
    ax.tick_params(axis="both", labelsize=DEFAULT_PLOT_CONFIG["tick_labelsize"])
    if DEFAULT_PLOT_CONFIG["tight_layout"]:
        plt.tight_layout()


class PlotFactory:
    """Factory class for generating DSBF-compliant plots."""

    @staticmethod
    def _is_empty(data: Union[pd.Series, pd.DataFrame]) -> bool:
        return data.empty if isinstance(data, pd.DataFrame) else data.size == 0

    @staticmethod
    def plot_histogram_static(
        series: pd.Series,
        save_path: str,
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        if PlotFactory._is_empty(series):
            return {
                "path": Path(save_path),
                "plot_data": {
                    "type": "histogram",
                    "data": {},
                    "config": {},
                    "annotations": ["Empty series"],
                },
            }

        x_label_str = str(series.name) if series.name else "Value"
        title_str = title or "Histogram"

        fig, ax = plt.subplots(figsize=DEFAULT_PLOT_CONFIG["figsize"])
        df = pd.DataFrame({x_label_str: series})
        sns.histplot(
            data=df,
            x=x_label_str,
            bins=30,
            kde=False,
            color=DEFAULT_PLOT_CONFIG["color"],
        )
        apply_static_style(ax, title_str, x_label_str, "Count")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

        return {
            "path": Path(save_path),
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
    ) -> PlotData:
        if PlotFactory._is_empty(series):
            return {
                "type": "histogram",
                "data": {},
                "config": {},
                "annotations": ["Empty series"],
            }

        x_label_str = str(series.name) if series.name else "Value"
        title_str = title or "Histogram"

        fig = go.Figure(
            [
                go.Histogram(
                    x=series, nbinsx=30, marker=dict(color=DEFAULT_PLOT_CONFIG["color"])
                )
            ]
        )
        fig.update_layout(
            title=title_str,
            xaxis_title=x_label_str,
            yaxis_title="Count",
            font=dict(family=DEFAULT_PLOT_CONFIG["font"]),
        )

        return {
            "type": "histogram",
            "data": {"x": series.tolist()},
            "config": {"title": title_str, "x_label": x_label_str, "y_label": "Count"},
            "annotations": annotations or [],
        }

    @staticmethod
    def plot_boxplot_static(
        series: pd.Series,
        save_path: str,
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        if PlotFactory._is_empty(series):
            return {
                "path": Path(save_path),
                "plot_data": {
                    "type": "boxplot",
                    "data": {},
                    "config": {},
                    "annotations": ["Empty series"],
                },
            }

        x_label_str = str(series.name) if series.name else "Value"
        title_str = title or "Boxplot"

        fig, ax = plt.subplots(figsize=DEFAULT_PLOT_CONFIG["figsize"])
        sns.boxplot(
            x=series,
            color=DEFAULT_PLOT_CONFIG["color"],
            ax=ax,
            orientation="horizontal",
        )  # type: ignore
        apply_static_style(ax, title_str, x_label_str, "")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

        return {
            "path": Path(save_path),
            "plot_data": {
                "type": "boxplot",
                "data": {"x": series.tolist()},
                "config": {"title": title_str, "x_label": x_label_str, "y_label": ""},
                "annotations": annotations or [],
            },
        }

    @staticmethod
    def plot_boxplot_interactive(
        series: pd.Series,
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
    ) -> PlotData:
        if PlotFactory._is_empty(series):
            return {
                "type": "boxplot",
                "data": {},
                "config": {},
                "annotations": ["Empty series"],
            }

        x_label_str = str(series.name) if series.name else "Value"
        title_str = title or "Boxplot"

        fig = go.Figure(
            [go.Box(x=series, marker=dict(color=DEFAULT_PLOT_CONFIG["color"]))]
        )
        fig.update_layout(
            title=title_str,
            xaxis_title=x_label_str,
            font=dict(family=DEFAULT_PLOT_CONFIG["font"]),
        )

        return {
            "type": "boxplot",
            "data": {"x": series.tolist()},
            "config": {"title": title_str, "x_label": x_label_str, "y_label": ""},
            "annotations": annotations or [],
        }

    @staticmethod
    def plot_barplot_static(
        series: pd.Series,
        save_path: str,
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        if PlotFactory._is_empty(series):
            return {
                "path": Path(save_path),
                "plot_data": {
                    "type": "bar",
                    "data": {},
                    "config": {},
                    "annotations": ["Empty series"],
                },
            }

        counts = series.value_counts()
        x_label_str = str(series.name) if series.name else "Category"
        title_str = title or "Bar Plot"

        fig, ax = plt.subplots(figsize=DEFAULT_PLOT_CONFIG["figsize"])
        counts.plot(kind="bar", color=DEFAULT_PLOT_CONFIG["color"], ax=ax)
        apply_static_style(ax, title_str, x_label_str, "Count")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

        return {
            "path": Path(save_path),
            "plot_data": {
                "type": "bar",
                "data": {"x": counts.index.tolist(), "y": counts.tolist()},
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
        title: Optional[str] = None,
        annotations: Optional[list[str]] = None,
    ) -> PlotData:
        if PlotFactory._is_empty(series):
            return {
                "type": "bar",
                "data": {},
                "config": {},
                "annotations": ["Empty series"],
            }

        counts = series.value_counts()
        x_label_str = str(series.name) if series.name else "Category"
        title_str = title or "Bar Plot"

        fig = go.Figure(
            [
                go.Bar(
                    x=counts.index.tolist(),
                    y=counts.tolist(),
                    marker=dict(color=DEFAULT_PLOT_CONFIG["color"]),
                )
            ]
        )
        fig.update_layout(title=title_str, xaxis_title=x_label_str, yaxis_title="Count")

        return {
            "type": "bar",
            "data": {"x": counts.index.tolist(), "y": counts.tolist()},
            "config": {"title": title_str, "x_label": x_label_str, "y_label": "Count"},
            "annotations": annotations or [],
        }

    @staticmethod
    def plot_null_matrix_static(
        df: pd.DataFrame,
        save_path: str,
        title: Optional[str] = "Null Matrix",
        annotations: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
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
    ) -> PlotData:
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

        return {
            "type": "matrix",
            "data": {},
            "config": {"title": title_str},
            "annotations": annotations or [],
        }

    @staticmethod
    def plot_correlation_static(
        df: pd.DataFrame,
        save_path: str,
        title: Optional[str] = "Correlation Matrix",
        annotations: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        corr = df.corr(numeric_only=True)
        if corr.empty:
            return {
                "path": Path(save_path),
                "plot_data": {
                    "type": "correlation",
                    "data": {},
                    "config": {},
                    "annotations": ["No numeric columns"],
                },
            }

        title_str = title or "Correlation Matrix"
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title(title_str)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

        return {
            "path": Path(save_path),
            "plot_data": {
                "type": "correlation",
                "data": corr.to_dict(),
                "config": {"title": title_str},
                "annotations": annotations or [],
            },
        }

    @staticmethod
    def plot_correlation_interactive(
        df: pd.DataFrame,
        title: Optional[str] = "Correlation Matrix",
        annotations: Optional[list[str]] = None,
    ) -> PlotData:
        corr = df.corr(numeric_only=True)
        if corr.empty:
            return {
                "type": "correlation",
                "data": {},
                "config": {},
                "annotations": ["No numeric columns"],
            }

        title_str = title or "Correlation Matrix"
        fig = go.Figure([go.Heatmap(z=corr.values, colorscale="RdBu")])
        fig.update_layout(title=title_str)

        return {
            "type": "correlation",
            "data": {},
            "config": {"title": title_str},
            "annotations": annotations or [],
        }

    @staticmethod
    def plot_missingness_matrix(
        df: pd.DataFrame,
        save_path: str,
        title: Optional[str] = "Missingness Matrix",
        annotations: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        Uses missingno to generate a missingness matrix plot. Returns static path only.

        Args:
            df (pd.DataFrame): DataFrame to visualize.
            save_path (str): Path to save static image.
            title (Optional[str]): Optional title (not directly used by missingno).
            annotations (Optional[list[str]]):

        Returns:
            dict: Dict with "path" key pointing to saved image.
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
        msno.matrix(df)
        plt.title(title_str)
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
