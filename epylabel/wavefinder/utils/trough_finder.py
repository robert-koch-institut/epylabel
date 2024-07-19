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
    trough_finder

DESCRIPTION
    This module provides a function which is used in wavefinder to locate troughs between peaks as well as after the
    final peak

FUNCTIONS
    run
"""

import pandas as pd
from pandas import DataFrame, Series


def run(
    peak_list: DataFrame,
    trough_list: DataFrame,
    raw_data: Series,
    prominence_threshold: float,
    prominence_height_threshold: float,
) -> DataFrame:
    """
    Locates the deepest troughs in raw_data between and possibly after the peaks in peak_list from candidates in
    trough_list.

    Parameters:
        peak_list (DataFrame): The set of peaks in the time series.
        trough_list (DataFrame): A set of troughs in the time series, one of which is to be retained between each
        consecutive pair of peaks.
        raw_data (Series): The time series
        prominence_threshold (float): The minimum prominence which a wave must have. Used for adding a final trough.
        proportional_prominence_threshold (float): The minimum prominence which a peak must have, as a ratio of the
        value at the peak. Used for adding a final trough.

    Returns:
        run(peak_list, trough_list, raw_data, prominence_threshold, proportional_prominence_threshold): A DataFrame
        containing the final list of peaks and troughs.
    """
    peak_list = peak_list.sort_values(by="location").reset_index(drop=True)
    results = peak_list
    for i in peak_list.index:
        if i < max(peak_list.index):
            wave_start = int(peak_list.loc[i, "location"])
            wave_end = int(peak_list.loc[i + 1, "location"])
            candidate_troughs = trough_list[
                (trough_list["location"] >= wave_start)
                & (trough_list["location"] <= wave_end)
            ]
            if len(candidate_troughs) > 0:
                candidate_troughs = candidate_troughs.loc[
                    candidate_troughs.idxmin()["y_position"]
                ]
                results = pd.concat([results, candidate_troughs.to_frame().T], axis=0)

                # try:
                #    results = pd.concat([results, candidate_troughs.to_frame().T], axis=0)
                # except:
                #    import pdb
                #    pdb.set_trace()

            else:
                trough_idx = raw_data.iloc[wave_start:wave_end].idxmin()["y_position"]
                candidate_troughs = DataFrame(
                    [[trough_idx], [raw_data.iloc[trough_idx]]],
                    columns=["location", "y_position"],
                )
                results = pd.concat([results, candidate_troughs.to_frame().T], axis=0)
    results = results.sort_values(by="location").reset_index(drop=True)

    # add final trough after final peak
    if len(peak_list) > 0:
        candidate_troughs = trough_list[
            trough_list.location >= peak_list.location.iloc[-1]
        ]
        if len(candidate_troughs) > 0:
            candidate_troughs = candidate_troughs.loc[
                candidate_troughs.idxmin()["y_position"]
            ]
            final_maximum = max(raw_data[(raw_data.index > candidate_troughs.location)])
            if (
                candidate_troughs.y_position
                <= (1 - prominence_height_threshold) * peak_list.y_position.iloc[-1]
            ) and (
                peak_list.y_position.iloc[-1] - candidate_troughs.y_position
                >= prominence_threshold
            ):
                if (
                    candidate_troughs.y_position
                    <= (1 - prominence_height_threshold) * final_maximum
                ) and (
                    final_maximum - candidate_troughs.y_position >= prominence_threshold
                ):
                    results = pd.concat(
                        [results, candidate_troughs.to_frame().T], axis=0
                    )

    return results
