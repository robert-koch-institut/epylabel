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
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .subalgorithms import algorithm_a as algorithm_a
from .subalgorithms import algorithm_b as algorithm_b
from .subalgorithms import algorithm_c_and_d as algorithm_c_and_d
from .subalgorithms import algorithm_init as algorithm_init


class WaveList:
    """
    NAME
        WaveList

    DESCRIPTION
        A WaveList object holds a time series, the parameters used to find waves
        in the time series, and the waves identified at different stages of the
        algorithm.

    ATTRIBUTES
        raw_data (Series): The original data from which the peaks and troughs
        are identified.
        series_name (str): The name of the series, for labelling plots.
        t_sep_a (int): Threshold specifying minimum wave duration.
        prominence_threshold (float): The minimum prominence which a wave must have.
        proportional_prominence_threshold (float): The minimum prominence which a
        peak must have, as a ratio of the value at the peak.
        peaks_initial (DataFrame): The list of peaks and troughs in raw_data.
        peaks_sub_a (DataFrame): The list of peaks and troughs after Sub-Algorithm A
        has merged short waves.
        peaks_sub_b (DataFrame): The list of peaks and troughs after Sub-Algorithm B
        has merged short transient features.
        peaks_sub_c (DataFrame): The list of peaks and troughs after Sub-Algorithms C
        and D have merged less prominent waves.
        peaks_cross_validated (DataFrame): The list of peaks and troughs after
        cross-validation.

    PROPERTIES
        waves (DataFrame): An alias for peaks_cross_validated if calculated, else
        peaks_sub_c, for better access to the final results.
            Index: RangeIndex
            Columns:
                location: The index of the peak or trough within raw_data.
                y_position: The value of the peak or trough within raw_data.
                prominence: The prominence of the peak or trough (as calculated
                with respect to other peaks and troughs, not with resepct to all
                of raw_data).
                peak_ind: 0 for a trough, 1 for a peak.

    METHODS
        __init__: After setting the parameters, this calls run to immediately
        execute the algorithm.
        waves: Gets the most recent list of peaks and troughs
        run: Finds the list of peaks and troughs in raw_data, then calls the
        Sub-Algorithms A, B, C and D to find the
        waves.
        cross_validate: Imputes the presence of additional waves in the from
        those in a second wavelist.
    """

    def __init__(
        self,
        raw_data: pd.Series,
        series_name: str,
        t_sep_a: int,
        prominence_threshold: float,
        proportional_prominence_threshold: float,
    ):
        """
        Creates the WaveList object and calls run to find the waves using the
        set parameters

        Parameters:
            raw_data (Series): The original data from which the peaks and troughs
            are identified.
            series_name (str): The name of the series, for labelling plots.
            t_sep_a (int): Threshold specifying minimum wave duration.
            prominence_threshold (float): The minimum prominence which a wave must
            have.
            proportional_prominence_threshold (float): The minimum prominence which
            a peak must have, as a ratio of the
            value at the peak.
        """
        # input data
        self.raw_data = raw_data
        self.series_name = series_name

        # configuration parameters
        self.t_sep_a = t_sep_a
        self.prominence_threshold = prominence_threshold
        self.proportional_prominence_threshold = proportional_prominence_threshold

        # peaks and troughs of waves are calculated by run()
        (
            self.peaks_initial,
            self.peaks_sub_a,
            self.peaks_sub_b,
            self.peaks_sub_c,
        ) = self.run()
        self.waves = self.get_waves()

    def get_waves(self) -> np.ndarray:
        """Provides the list of waves, peaks_sub_c"""
        # location, prominence, y_position, peak_ind
        signals_c = self.peaks_sub_c[["location", "peak_ind"]].values.astype(int)
        signals_b = self.peaks_sub_b[["location", "peak_ind"]].values.astype(int)
        wave_signals = signals_c.copy()

        for i in range(len(signals_c)):
            if signals_c[i, 1] == 1:
                previous_trough = np.max(
                    signals_b[
                        (signals_b[:, 0] < signals_c[i, 0]) & (signals_b[:, 1] == 0)
                    ],
                    initial=-1,
                )
                if previous_trough == -1 or previous_trough == signals_c[i - 1, 0]:
                    continue
                else:
                    # insert trough before peak
                    wave_signals = np.insert(signals_c, i, [previous_trough, 0], axis=0)

        if wave_signals[0, 1] == 1:
            # prepend pink_ind 0 at first lkdata.index[0]
            wave_signals = np.vstack((np.array([0, 0]), wave_signals))

        if wave_signals[-1, 1] == 0:
            # append peak_ind 1 at last lkdata.index[-1]
            wave_signals = np.vstack(
                (wave_signals, np.array([self.raw_data.index.max(), 1]))
            )

        return wave_signals

    def run(
        self,
        prominence_threshold: float = 0.1,
        proportional_prominence_threshold: float = 0.05,
        t_sep_a: float = 20,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,]:
        """
        Executes the algorithm by finding the initial list of peaks and troughs,
        then calling A through D.
        """
        peaks_initial, prominence_updater = algorithm_init.run(self.raw_data)

        peaks_sub_a = algorithm_a.run(
            input_data_df=peaks_initial,
            prominence_updater=prominence_updater,
            t_sep_a=self.t_sep_a,
        )

        peaks_sub_b = algorithm_b.run(
            raw_data=self.raw_data,
            input_data_df=peaks_sub_a,
            prominence_updater=prominence_updater,
            t_sep_a=self.t_sep_a,
        )

        peaks_sub_c = algorithm_c_and_d.run(
            raw_data=self.raw_data,
            input_data_df=peaks_sub_b,
            prominence_threshold=self.prominence_threshold,
            proportional_prominence_threshold=self.proportional_prominence_threshold,
        )

        return peaks_initial, peaks_sub_a, peaks_sub_b, peaks_sub_c
