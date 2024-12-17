from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def boxplot_electricity_consumption(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    xticklabel: List | None,
    hue: str | None,
    colours: int = 2,
    is_legend: bool = False,
):
    """
    Creates a boxplot for electricity consumption data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to plot.
    x (str): The column name to be used for the x-axis.
    y (str): The column name to be used for the y-axis.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    xticklabel (List | None): Custom labels for the x-ticks.
    If None, default labels are used.
    hue (str | None): The column name to be used for color encoding.
    If None, no hue is applied.
    colours (int): The number of colors to use in the palette. Default is 2.
    is_legend (bool): Whether to display the legend. Default is False.

    Returns:
    None
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    palette = sns.color_palette("husl", colours)
    sns.boxplot(x=x, y=y, hue=hue, legend=is_legend, data=df, palette=palette)
    # Reduce the frequency of the xticks and change the labels to be in
    # the range [0,24] hours
    if xlabel == "Hour":
        ax.set_xticks(range(1, 49, 2))
    if xticklabel:
        ax.set_xticklabels(xticklabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()


def plot_week_consumption(
    df: pd.DataFrame, week_from: str, week_to: str, title: str, xlabel: str, ylabel: str
):
    """
    Plots the electricity consumption data for a week.
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    Returns:
        None
    """
    df.loc[(df.index > week_from) & (df.index < week_to)]["tsd"].plot(
        figsize=(15, 5), xlabel=xlabel, ylabel=ylabel, title=title
    )
