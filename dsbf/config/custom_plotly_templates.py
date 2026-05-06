# dsbf/config/custom_plotly_templates.py

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def register_dark_theme() -> None:
    dark_color = "#ffffff"
    dark_background = "#292929"
    grid_color = "#444444"
    outline_color = "#dcdcdc"
    # light_gray = "#cccccc"

    axis_common: dict[str, bool | str] = dict(
        showgrid=False,
        gridcolor=grid_color,
        linecolor=dark_color,
        ticks="outside",
        showline=True,
    )

    axis_common_no_title: dict[str, bool | str] = {
        k: v for k, v in axis_common.items() if k != "title"
    }

    shape_defaults: dict[str, dict[str, int] | float | str] = dict(
        fillcolor="white",
        line={"width": 0},
        opacity=0.3,
    )
    annotation_defaults: dict[str, int] = {"arrowhead": 0, "arrowwidth": 1}

    template = go.layout.Template(
        layout=dict(
            autotypenumbers="strict",
            colorway=px.colors.qualitative.D3,
            font=dict(color=dark_color),
            paper_bgcolor=dark_background,
            plot_bgcolor=dark_background,
            hovermode="closest",
            hoverlabel=dict(align="left"),
            title=dict(x=0.05),
            xaxis={
                **axis_common,
                "zeroline": False,
                "automargin": True,
                "zerolinecolor": outline_color,
            },
            yaxis={
                **axis_common,
                "zeroline": False,
                "automargin": True,
                "zerolinecolor": outline_color,
            },
            scene=dict(
                xaxis={
                    **axis_common,
                    "backgroundcolor": dark_background,
                    "gridwidth": 2,
                    "zeroline": False,
                },
                yaxis={
                    **axis_common,
                    "backgroundcolor": dark_background,
                    "gridwidth": 2,
                    "zeroline": False,
                },
                zaxis={
                    **axis_common,
                    "backgroundcolor": dark_background,
                    "gridwidth": 2,
                    "zeroline": False,
                },
            ),
            polar=dict(
                bgcolor=dark_background,
                angularaxis=axis_common_no_title,
                radialaxis=axis_common_no_title,
            ),
            ternary=dict(
                aaxis=axis_common_no_title,
                baxis=axis_common_no_title,
                caxis=axis_common_no_title,
            ),
            geo=dict(
                bgcolor=dark_background,
                landcolor=dark_background,
                subunitcolor=dark_background,
                showland=True,
                showlakes=True,
                lakecolor=dark_background,
            ),
            coloraxis=dict(
                colorbar=dict(outlinewidth=1, tickcolor=dark_color, ticks="outside")
            ),
            annotationdefaults=annotation_defaults,
            shapedefaults=shape_defaults,
            mapbox=dict(style="dark"),
        ),
        data=dict(
            bar=[dict(marker=dict(line=dict(color=dark_background, width=0.5)))],
            histogram=[dict(marker=dict(line=dict(color=dark_background, width=0.6)))],
            pie=[dict(automargin=True)],
        ),
    )

    pio.templates["simple_dark"] = template
