"""
Module for visualizing time series labels using Dash and Plotly.

This module provides functions for loading time series label data,
applying labeling algorithms, and creating interactive visualizations
using the Dash framework and Plotly library.

Functions:
    append_signal_bar(location_data, j)
    find_all_label_files(folder)
    load_labels(paths, engine="pyarrow", dtype_backend="pyarrow")
    filter_target(df, start, end)
    sample_data(locations, n_samples)
    standard_layout(fig, h, title)
    label_plot(data, fig, name, color)
    label_plot_facet(data, fig, r, c, name, color, show_name)
    start_app(path)

Classes:
    None

Public Constants:
    None

Public Module-Level Variables:
    None

Dependencies:
    datetime, random, pathlib.Path, typing.List, typing.Union, numpy,
    pandas, plotly.graph_objects, dash.Dash, dash.Input, dash.Output,
    dcc, html, plotly.subplots
"""
import datetime
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots


def append_signal_bar(location_data: pd.DataFrame, j: int) -> pd.DataFrame:
    """
    Append a 'signal_bar' column to the input DataFrame.

    :param location_data: The input DataFrame.
    :type location_data: pandas.DataFrame
    :param j: A coefficient for signal_bar calculation.
    :type j: int
    :return: The input DataFrame with the 'signal_bar' column added.
    :rtype: pd.DataFrame
    """
    location_data["signal_float"] = location_data["signal"].astype("float32[pyarrow]")
    location_data.eval("signal_bar = signal_float * value.max()", inplace=True)
    location_data["signal_bar"] *= 1 + j / 100
    return location_data.drop("signal_float", axis=1)


def find_all_label_files(folder) -> dict:
    """
    Gets a list of paths to all .parquet files in folder. recursively
    :param folder:
    :return: list
    """
    return {p.stem: p for p in Path(folder).rglob("*.parquet")}


def load_labels_helper(data: pd.DataFrame, historic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Called withing load_labels for each labelling file to be merged with the historic
    data Currently the labels data only holds labels. The values are joined to the data
    here so that later a proper signal bar can be calculated for the vizualiation.

    :param data: pd.DataFrame
    :param historic_data: pd.DataFrame
    :return: pd.DataFrame
    """
    return data.rename(columns={"label": "signal"}).merge(
        historic_data, how="outer", on=["target", "location"]
    )


# label_file


def load_labels(
    paths, historic_data, engine="pyarrow", dtype_backend="pyarrow"
) -> dict:
    """
    Given a list of paths loads the parquet file
    :param paths: list of paths
    :return: dict
    """
    return {
        p.stem: load_labels_helper(
            pd.read_parquet(
                p,
                columns=["target", "location", "label"],
                engine=engine,
                dtype_backend=dtype_backend,
            ),
            historic_data,
        )
        for p in paths
    }


def filter_target(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """
    Filter a DataFrame based on a target range.

    :param df: The input DataFrame.
    :type df: pandas.DataFrame
    :param start: The lower bound of the target range (pandas datetime).
    :type start: pandas.Timestamp
    :param end: The upper bound of the target range (pandas datetime).
    :type end: pandas.Timestamp
    :return: A filtered DataFrame containing rows with target dates
    within the specified range.
    :rtype: pandas.DataFrame
    """
    return df[(df["target"] >= start) & (df["target"] <= end)]


def sample_data(locations: List[str], n_samples: int) -> List[str]:
    """
    Sample data from a list of locations.

    :param locations: A list of location names.
    :type locations: list of str
    :param n_samples: The number of samples to generate.
    :type n_samples: int
    :return: A list of sampled data.
    :rtype: list of str
    """
    sampled_locations = random.sample(locations, n_samples)
    return sampled_locations


def standard_layout(fig: go.Figure, h: Union[int, float], title: str) -> go.Figure:
    """
    Apply a standard layout to a Plotly figure.

    :param fig: The Plotly figure to which the layout will be applied.
    :type fig: plotly.graph_objs.Figure
    :param h: Height of the figure.
    :type h: Union[int, float]
    :param title: Title of the figure.
    :type title: str
    :return: The Plotly figure with the updated layout.
    :rtype: plotly.graph_objs.Figure
    """
    fig.update_layout(
        height=h,
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
        margin=dict(l=25, r=25, b=25, t=25, pad=4),
        title_text=title,
        plot_bgcolor="white",
    )
    return fig


def label_plot(
    data: pd.DataFrame, fig: go.Figure, name: str, color: str, add_line: bool
) -> go.Figure:
    """
    Add labeled plots to a Plotly figure.

    :param data: The data to plot.
    :type data: pandas.DataFrame
    :param fig: The Plotly figure to which traces will be added.
    :type fig: plotly.graph_objs.Figure
    :param name: Name of the label.
    :type name: str
    :param color: Color of the label.
    :type color: str
    :return: The updated Plotly figure.
    :rtype: plotly.graph_objs.Figure
    """
    linecolor = color.rsplit(",", 1)[0] + ", 1)"
    if add_line:
        fig.add_trace(
            go.Scatter(
                x=data["target"].values,
                y=data["value"].values,
                line=dict(color="black"),
                name="value",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=data["target"].values,
            y=data["signal_bar"].values,
            fill="tozeroy",
            mode=None,
            fillcolor=color,
            line=dict(color=linecolor),
            line_shape="hv",
            name=name,
        )
    )
    return fig


def label_plot_facet(
    data: pd.DataFrame,
    fig: go.Figure,
    r: int,
    c: int,
    name: str,
    color: str,
    show_name: bool,
    add_line: bool,
) -> go.Figure:
    """
    Add labeled plots for individual facets in a facet plot.

    :param data: The data to plot.
    :type data: pandas.DataFrame
    :param fig: The Plotly figure to which traces will be added.
    :type fig: plotly.graph_objs.Figure
    :param r: Row index of the facet.
    :type r: int
    :param c: Column index of the facet.
    :type c: int
    :param name: Name of the label.
    :type name: str
    :param color: Color of the label.
    :type color: str
    :param show_name: Whether to show the label name in the legend.
    :type show_name: bool
    :return: The updated Plotly figure.
    :rtype: plotly.graph_objs.Figure
    """
    linecolor = color.rsplit(",", 1)[0] + ", 1)"
    if add_line:
        fig.add_trace(
            go.Scatter(
                x=data["target"].values,
                y=data["value"].values,
                line=dict(color="black"),
                legendgroup="value",
                name="value",
                showlegend=show_name,
            ),
            row=r,
            col=c,
        )
    fig.add_trace(
        go.Scatter(
            x=data["target"].values,
            y=data["signal_bar"].values,
            fill="tozeroy",
            mode=None,
            fillcolor=color,
            line=dict(color=linecolor),
            line_shape="hv",
            legendgroup=name,
            name=name,
            showlegend=show_name,
        ),
        row=r,
        col=c,
    )
    return fig


def start_app(path: str, path_historic: str):
    """
    Start a Dash app that visualizes labels for various time series.

    :param path: Path to a directory that stores label files in the
    standardized long format.
    :type path: str
    """
    app = Dash(__name__)

    pd.options.mode.chained_assignment = None
    alpha = 0.1
    colors = [
        f"rgba(255, 0, 0, {alpha})",
        f"rgba(0, 255, 0, {alpha})",
        f"rgba(0, 0, 255, {alpha})",
        f"rgba(255, 255, 0, {alpha})",
        f"rgba(0, 255, 255, {alpha})",
    ]

    label_files = find_all_label_files(path)
    label_file_choices = [
        {"label": name, "value": data.absolute().as_posix()}
        for name, data in label_files.items()
    ]

    # nur für den Start, könnte entfernt werden
    # da der Filter für die Locations inzwischen dynamisch
    # gesetzt wird
    locations_set = set()
    start_dt = datetime.datetime.now()
    end_dt = datetime.datetime(1900, 1, 1)
    for label_path in label_files.values():
        label_locs_tar = pd.read_parquet(
            label_path,
            columns=["location", "target"],
            engine="pyarrow",
            dtype_backend="pyarrow",
        )
        locations_set.update(label_locs_tar.location.unique())
        start_dt = min(start_dt, label_locs_tar.target.min())
        end_dt = max(end_dt, label_locs_tar.target.max())
    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    locations = list(locations_set)
    label_file_default = random.sample(label_file_choices, 1)[0]["value"]
    locations_default = random.sample(locations, 1)[0]
    historic_data = pd.read_parquet(
        path_historic,
        columns=["target", "location", "value"],
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    app.layout = html.Div(
        children=[
            dcc.Dropdown(
                options=label_file_choices,
                value=[label_file_default],
                multi=True,
                id="file-select",
            ),
            dcc.Dropdown(
                options=locations,
                value=locations_default,
                multi=False,
                id="location-select",
            ),
            dcc.DatePickerRange(
                id="daterange",
                min_date_allowed=start,
                max_date_allowed=end,
                start_date=start,
                end_date=end,
            ),
            html.Div(style={"height": "20px"}),
            html.Button("Sample", id="sample-button", n_clicks=0),
            html.Div(style={"height": "20px"}),
            dcc.Graph(id="facet-plot", figure={}),
            dcc.Graph(id="single-plot", figure={}),
        ]
    )

    @app.callback(
        Output("facet-plot", "figure"),
        Output("single-plot", "figure"),
        Input("file-select", "value"),
        Input("location-select", "value"),
        Input("sample-button", "n_clicks"),
        Input("daterange", "start_date"),
        Input("daterange", "end_date"),
    )
    def update_plots(selected_file, selected_location, clicked, start, end):
        """

        :param selected_file:
        :param selected_location:
        :param clicked:
        :param locations:
        :return:
        """
        data = load_labels([Path(p) for p in selected_file], historic_data)
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        data = {k: filter_target(v, start, end) for k, v in data.items()}
        data = {name: d.groupby("location") for name, d in data.items()}
        fig = go.Figure()
        fig = standard_layout(fig, 800, selected_location)
        j = 0
        # Einzelner großer Plot
        for name, d in data.items():
            location_data = d.get_group(selected_location)
            location_data = append_signal_bar(location_data, j)
            fig = label_plot(location_data, fig, name, colors[j], j == 0)
            j += 1
        # Facet Plot Update
        try:
            sampled_locations = sample_data(locations, 9)
            sampled_locations = sorted(sampled_locations)
            facets = make_subplots(
                rows=3,
                cols=3,
                subplot_titles=[str(c) for c in sampled_locations],
                horizontal_spacing=0.05,
                vertical_spacing=0.05,
            )
            facets = standard_layout(facets, 1400, "")
            ids = np.arange(9).reshape(3, 3)
            for i in range(9):
                r = np.where(ids == i)[0][0] + 1
                c = np.where(ids == i)[1][0] + 1
                selected_location_facet = sampled_locations[i]
                j = 0
                for name, d in data.items():
                    location_data = d.get_group(selected_location_facet)
                    location_data = append_signal_bar(location_data, j)
                    facets = label_plot_facet(
                        location_data, facets, r, c, name, colors[j], i == 0, j == 0
                    )
                    j += 1
        except ValueError:
            # case when only one timeseries available
            facets = {}
        return fig, facets

    app.run_server(debug=True)


start_app("output/LK/labels", "output/LK/incidence.parquet")
