"""
Original Source:
This file is taken from the repository: https://github.com/covid19db/epidemiological-waves
Original Authors: John Harvey
Original Repository: epidemiological-waves

Associated Paper:
Epidemiological waves - Types, drivers and modulators in the COVID-19 pandemic
Authors: Harvey et al.
Journal: Heliyon, Volume 9, Issue 5
Year: 2023
DOI: https://doi.org/10.1016/j.heliyon.2023.e16015

NAME
    wavefinder

DESCRIPTION
    A Python package to identify waves in time series.
    ==================================================

    wavefinder provides two classes and two associated plotting functions. WaveList implements an algorithm to
    identify the waves in time series, which can be plotted using plot_peaks. It also implements an
    algorithm to impute additional waves from a reference WaveList object,
    which plot_cross_validator plots.

PACKAGE CONTENTS
    WaveList
    plot_peaks
    plot_cross_validator
"""

from .wavelist import WaveList

__all__ = ["WaveList"]
