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
    algorithm_init

DESCRIPTION
    This module provides an initial list of peaks and troughs along with the
    ProminenceUpdater to WaveList.

FUNCTIONS
    run
    init_peaks_and_troughs
"""

import numpy as np
from pandas import DataFrame
from scipy.signal import find_peaks

from ..utils.prominence_updater import ProminenceUpdater


def init_peaks_and_troughs(data):
    """
    Identifies an initial list of peaks and troughs using scipy.signal.find_peaks.

    Parameters:
        data (Series): The original data from which the peaks and troughs are to
        be identified.

    Returns:
        init_peaks_and_troughs(data): The list of all peaks and troughs in data,
        with their location, prominence and value.
    """
    peak = find_peaks(data.values, prominence=0, distance=1)
    trough = find_peaks([-x for x in data.values], prominence=0, distance=1)

    # collect into a single dataframe
    df = DataFrame(
        data=np.transpose(
            [
                np.append(data.index[peak[0]], data.index[trough[0]]),
                np.append(peak[1]["prominences"], trough[1]["prominences"]),
            ]
        ),
        columns=["location", "prominence"],
    )
    df["peak_ind"] = np.append([1] * len(peak[0]), [0] * len(trough[0]))
    df.loc[:, "y_position"] = data[df["location"]].values
    df = df.sort_values(by="location").reset_index(drop=True)
    return df


def run(data):
    """
    Identifies the peaks and troughs and initialises the ProminenceUpdater.

    Parameters:
        data (Series): The original data from which the peaks and troughs are to
        be identified.

    Returns:
        run(data): A tuple made up of the DataFrame of peaks and troughs and the
        ProminenceUpdater for data.
    """
    if len(data) == 0:
        return data, None, None
    pre_algo = init_peaks_and_troughs(data)
    prominence_updater = ProminenceUpdater(data)
    return pre_algo, prominence_updater
