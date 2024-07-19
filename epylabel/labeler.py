"""
A labelling algorithm is implemented as a Transformation. The Transformation
is the interface that defines the abstract method transform that returns a
pd.DataFrame. For most transformations it should be enforced, that the input
format of the data (wide format) is the same as the output format. Adhering to
this the user can chain multiple transformations like a log transformation,
labelling using the Shapelet, GapFilling and the calculation of summary statistics.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from scipy.stats import linregress

import epylabel.wavefinder as wf
from epylabel.metrics import summary
from epylabel.utils import assert_same_structure_list, to_long


class Transformation(ABC):
    """
    Interface of the major building blocks of a labeling process in epylabel.

    This class defines an abstract interface for performing transformations
    on data during the labeling process.

    Attributes:
        None

    Methods:
        transform(data: pd.DataFrame) -> pd.DataFrame:
            Abstract method to perform a transformation on the input data.

    """

    @abstractmethod
    def transform(self, data) -> pd.DataFrame:
        """
        Abstract method to perform a transformation on the input data. The input
        data has to be in the predefined format of a pd.DataFrame. Each column
        representing the timeseries of a location or geographical entity.

        This method should be implemented by subclasses to perform specific
        transformations on the given DataFrame.

        :return: Transformed data after applying the specified transformation.
        :rtype: pd.DataFrame
        """
        pass


class LogTransform(Transformation):
    """
    Transformation class for applying a logarithmic transformation to data.

    This class inherits from the Transformation interface and provides a method
    to apply a logarithmic transformation to the input DataFrame.

    Attributes:
        epsilon (float): A small positive constant added to the input data
        to handle non-positive values.

    Methods:
        transform(data: pd.DataFrame) -> pd.DataFrame:
            Applies a logarithmic transformation to the input data after adding epsilon.
    """

    def __init__(self, epsilon: float = 1e-6):
        """
        Initializes a LogTransform object.

        :param epsilon: A small positive constant added to the input
        data to handle non-positive values.
        :type epsilon: float, optional
        """
        self.epsilon = epsilon

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a logarithmic transformation to the input data.

        This method adds the epsilon value to the input data and then applies
        the natural logarithm transformation element-wise.

        :param data: The input data on which the transformation is to be applied.
        :type data: pd.DataFrame
        :return: Transformed data after applying the logarithmic transformation.
        :rtype: pd.DataFrame
        """
        return np.log(data + self.epsilon)


class ExponentialGrowth(Transformation):
    """
    Transformation class for applying an exponential growth transformation on data.

    This class inherits from the Transformation interface and provides a method to
    apply an exponential growth transformation to the input DataFrame.

    Attributes:
        n_days (int): Number of days for rolling window calculation.
        thresh (float): Threshold value for the transformation.
        log_transform (bool): Flag to indicate whether to perform a log transformation
        before applying.
        epsilon (float): A small positive constant added to the data when applying
        log transformation.

    Methods:
        transform(data: pd.DataFrame) -> pd.DataFrame:
            Applies the exponential growth transformation on the input data.
        slope_p(x: np.ndarray, n: int) -> float:
            Calculates the slope p-value using linear regression.
    """

    def __init__(
        self,
        n_days: int,
        thresh: float,
        log_transform: bool = False,
        epsilon: float = 1e-6,
    ):
        """
        Initializes an ExponentialGrowth object.

        :param n_days: Number of days for rolling window calculation.
        :type n_days: int
        :param thresh: Threshold value for the transformation.
        :type thresh: float
        :param log_transform: Flag to indicate whether to perform a log
        transformation before applying.
        :type log_transform: bool, optional
        :param epsilon: A small positive constant added to the data when
        applying log transformation.
        :type epsilon: float, optional
        """
        self.n_days = n_days
        self.thresh = thresh
        self.log_transform = log_transform
        self.epsilon = epsilon

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the exponential growth transformation on the input data.

        :param data: The input data.
        :type data: pd.DataFrame
        :return: Transformed data with True/False values based on the transformation.
        :rtype: pd.DataFrame
        """
        if self.log_transform:
            lt = LogTransform(self.epsilon)
            data = lt.transform(data)

        def f(x):
            return self.slope_p(x, self.n_days)

        p_vals = data.rolling(window=self.n_days, center=True).apply(f)
        return p_vals > self.thresh

    def slope_p(self, x: np.ndarray, n: int) -> float:
        """
        Calculates the slope p-value using linear regression.

        :param x: Input data for which slope p-value is calculated.
        :type x: np.ndarray
        :param n: Number of data points.
        :type n: int
        :return: Calculated slope p-value.
        :rtype: float
        """
        model = linregress(np.arange(n), x, alternative="greater")
        return 1 - model.pvalue


class Shapelet(Transformation):
    """
    Transformation class for applying a shapelet-based transformation on data.

    This class inherits from the Transformation interface and provides methods
    to apply a shapelet-based transformation
    to the input data.

    Attributes:
        n_days (int): Number of days for rolling window calculation.
        x_max (float): Maximum value for the x values used in the shapelet calculation.
        thresh (float): Threshold value for the transformation.

    Methods:
        transform(data: pd.DataFrame) -> pd.DataFrame:
            Applies the shapelet-based transformation on the input data.
        surge() -> np.ndarray:
            Generates surge values using the shapelet formula.
        pearson_r(x: np.ndarray, y: np.ndarray) -> float:
            Calculates the Pearson correlation coefficient between two arrays.
    """

    def __init__(self, n_days: int, x_max: float, thresh: float):
        """
        Initializes a Shapelet object.

        :param n_days: Number of days for rolling window calculation.
        :type n_days: int
        :param x_max: Maximum value for the x values used in the shapelet calculation.
        :type x_max: float
        :param thresh: Threshold value for the transformation.
        :type thresh: float
        """
        self.n_days = n_days
        self.x_max = x_max
        self.thresh = thresh

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the shapelet-based transformation on the input data.

        :param data: The input data.
        :type data: pd.DataFrame
        :return: Transformed data with True/False values based on the transformation.
        :rtype: pd.DataFrame
        """
        y = self.surge(self.n_days, self.x_max)

        def f(x):
            return self.pearson_r(x, y=y)

        r_vals = data.rolling(window=self.n_days, center=True).apply(f)
        return r_vals > self.thresh

    def surge(self, n_days: int, x_max: float) -> np.ndarray:
        """
        Generates surge values using the shapelet formula.

        :param n_days: Number of days for the surge values calculation.
        :type n_days: int
        :param x_max: Maximum value for the x values used in the shapelet calculation.
        :type x_max: float
        :return: Array of surge values based on the shapelet formula.
        :rtype: np.ndarray
        """
        return 2 ** (np.arange(n_days) / n_days * x_max)

    def pearson_r(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the Pearson correlation coefficient between two arrays.

        :param x: First input array.
        :type x: np.ndarray
        :param y: Second input array.
        :type y: np.ndarray
        :return: Calculated Pearson correlation coefficient.
        :rtype: float
        """
        try:
            model = linregress(x, y, alternative="greater")
            return model.rvalue
        except ValueError:
            # if x is all same value
            return 0


class Bcp(Transformation):
    """
    Transformation class for applying a Bayesian changepoint detection transformation
    using the bcp library.

    This class inherits from the Transformation interface and provides methods to
    apply a Bayesian changepoint detection transformation on the input data.

    Attributes:
        d (int): Maximum number of changepoints to consider.
        p0 (float): Prior probability of no changepoint.
        thresh (float): Threshold value for the transformation.

    Methods:
        changepoints(x: np.ndarray, d: int, p0: float) -> np.ndarray:
            Detects changepoints in the input data using Bayesian changepoint detection.
        transform(data: pd.DataFrame) -> pd.DataFrame:
            Applies the Bayesian changepoint detection transformation on the input data.
    """

    def __init__(self, d: int, p0: float, thresh: float):
        """
        Initializes a Bcp object.

        :param d: Maximum number of changepoints to consider.
        :type d: int
        :param p0: Prior probability of no changepoint.
        :type p0: float
        :param thresh: Threshold value for the transformation.
        :type thresh: float
        """
        try:
            self.bcp = rpackages.importr("bcp")
        except rpackages.PackageNotInstalledError:
            robjects.r("install.packages('bcp')")
            self.bcp = rpackages.importr("bcp")
        self.p0 = p0
        self.d = d
        self.thresh = thresh

    def changepoints(self, x: np.ndarray, d: int, p0: float) -> np.ndarray:
        """
        Detects changepoints in the input data using Bayesian changepoint detection.

        :param x: Input data for which changepoints are to be detected.
        :type x: np.ndarray
        :param d: Maximum number of changepoints to consider.
        :type d: int
        :param p0: Prior probability of no changepoint.
        :type p0: float
        :return: Array of detected changepoints.
        :rtype: np.ndarray
        """
        x_r = robjects.FloatVector(x.values)
        out = self.bcp.bcp(x_r, d=d, p0=p0)
        return np.array(out.rx2["posterior.mean"]).flatten()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the Bayesian changepoint detection transformation on the input data.

        :param data: The input data.
        :type data: pd.DataFrame
        :return: Transformed data with True/False values based on the transformation.
        :rtype: DataFrame
        """

        def f(x):
            return self.changepoints(x, d=self.d, p0=self.p0)

        labels = data.apply(f)
        return labels > self.thresh


class WaveFinder(Transformation):
    """
    Transformation class for finding waves in input data using specified thresholds.

    This class inherits from the Transformation interface and provides methods to
    find waves in input data based on the provided thresholds.

    Attributes:
        abs_prominence_threshold (float): Absolute prominence threshold for
        wave detection.
        prominence_height_threshold (float): Prominence height threshold
        for wave detection.
        t_sep_a (float): Time separation threshold for wave detection.

    Methods:
        find_waves(data: pd.DataFrame) -> np.ndarray:
            Finds waves in the input data based on the specified thresholds.
        transform(data: pd.DataFrame) -> pd.DataFrame:
            Applies the wave detection transformation on the input data.
    """

    def __init__(
        self,
        abs_prominence_threshold: float,
        prominence_height_threshold: float,
        t_sep_a: int,
    ):
        """
        Initializes a WaveFinder object.

        :param abs_prominence_threshold: Absolute prominence threshold
        for wave detection.
        :type abs_prominence_threshold: float
        :param rel_prominence_threshold: Relative prominence threshold
        for wave detection.
        :type prominence_height_threshold: float
        :param t_sep_a: Time separation threshold for wave detection.
        :type t_sep_a: int
        """
        self.abs_prominence_threshold = abs_prominence_threshold
        self.prominence_height_threshold = prominence_height_threshold
        self.t_sep_a = t_sep_a

    def find_waves(self, data: pd.DataFrame) -> np.ndarray:
        """
        Finds waves in the input data based on the specified thresholds.

        :param data: The input data containing target values for wave detection.
        :type data: pd.DataFrame
        :return: Array of labels indicating detected waves.
        :rtype: np.ndarray
        """
        data = data.reset_index().drop("target", axis=1)
        data_value = pd.Series(data.values.flatten())

        wavelist = wf.WaveList(
            data_value,
            "",
            self.t_sep_a,
            self.abs_prominence_threshold,
            self.prominence_height_threshold,
        )
        labels = np.zeros(len(data))
        wave_signals = wavelist.waves
        for i in range(wave_signals.shape[0]):
            # set signal to 1 for every wave
            if wave_signals[i, 1] == 1:
                # allows for multiple consecutive troughs
                labels[wave_signals[i - 1, 0] : wave_signals[i, 0]] = 1
        return labels

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the wave detection transformation on the input data.

        :param data: The input data.
        :type data: pd.DataFrame
        :return: Transformed data with Boolean values indicating detected waves.
        :rtype: pd.DataFrame
        """
        data = data.apply(self.find_waves)
        return data.astype(bool)


class GapFiller(Transformation):
    """
    Transformation class for applying a gap filling transformation on data. The
    data is a previously labelled pd.DataFrame with booleans indicating the
    presence or absence of a label/signal.

    This class inherits from the Transformation interface and provides methods
    to fill gaps in the input data.

    Attributes:
        n_days (int): Number of days for the gap filling window.

    Methods:
        transform(data: pd.DataFrame) -> pd.DataFrame:
            Applies the gap filling transformation on the input data.
        outbreak_ids(x: np.ndarray, n_days: int) -> np.ndarray:
            Fills gaps in the input array based on the outbreak condition.
    """

    def __init__(self, n_days: int):
        """
        Initializes a GapFiller object.

        :param n_days: Number of days for the gap filling window.
        :type n_days: int
        """
        self.n_days = n_days

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the gap filling transformation on the input data.
        The input data are labels that have been generated with any
        epylabel algortithm

        :param data: The input data.
        :type data: pd.DataFrame
        :return: Transformed data with True/False values based on the transformation.
        :rtype: DataFrame
        """

        def f(x):
            return self.outbreak_ids(x, n_days=self.n_days)

        data = data.apply(f)
        return data > 0

    def outbreak_ids(self, x: np.ndarray, n_days: int) -> np.ndarray:
        """
        Fills gaps in the input array based.

        :param x: Input array for which gaps are to be filled.
        :type x: np.ndarray
        :param n_days: Number of days for the gap filling window.
        :type n_days: int
        :return: Array with gaps filled based on labels and windowsize.
        :rtype: np.ndarray
        """
        for i in range(len(x)):
            if x[i] == 0:
                left_window = x[max(0, i - n_days) : i]
                right_window = x[i + 1 : min(len(x), i + n_days + 1)]
                if 1 in left_window and 1 in right_window:
                    x[i] = 1
        return x > 0


class Changerate:
    """
    A class to calculate the change rate of a rolling moving average over a
    specified number of days.

    Parameters:
    :param n_days: Number of days to calculate the rolling moving average. Default is 7.
    """

    def __init__(self, n_days: int = 7, changerate_ceiling: int = 6):
        """
        Initialize the Changerate object.

        :param n_days: Number of days to calculate the rolling moving average
        and its changerate. Default is 7.
        :param changerate_ceiling: Maximum value that the changerate can take. Set this
        value low if there is one big spike, e.g. at the beginning of the COVID-19
        pandemic. Set this value high if you have many spikes. Default is 6.
        """
        self.n_days = n_days
        self.changerate_ceiling = changerate_ceiling

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the change rate of a rolling moving average over the
        specified number of days.

        :param data: A pandas DataFrame containing the data for which to
        calculate the change rate.
        :return: A pandas DataFrame containing the calculated change rates.
        """
        moving_averages = data.rolling(self.n_days, min_periods=1).mean()
        changerate = (moving_averages / moving_averages.shift(self.n_days)).fillna(0)
        changerate[changerate > self.changerate_ceiling] = self.changerate_ceiling
        mask = (moving_averages >= 1) & (moving_averages.shift(self.n_days) >= 1)
        return (changerate * mask).fillna(0)

class Ensemble(Transformation):
    """
    Transformation class for creating an ensemble based on a minimum number of labels
    per timepoint.

    If more or equal than the minimum number of models suggest a label for a certain
    location and timepoint, a label will be suggested by the ensemble.

    This class inherits from the Transformation interface and provides methods to
    create an ensemble based on a minimum number of labels.

    Attributes:
        n_min (int): Minimum number of labels required for each ensemble element.

    Methods:
        transform(*args: List[pd.DataFrame]) -> pd.DataFrame:
            Applies the ensemble transformation on the provided input dataframes.
    """

    def __init__(self, n_min: int):
        """
        Initializes an Ensemble object.

        :param n_min: Minimum number of labels required for each ensemble element.
        :type n_min: int
        """
        self.n_min = n_min

    def transform(self, *args) -> pd.DataFrame:
        """
        Applies the ensemble transformation on the provided input dataframes.

        :param args: List of input dataframes to create an ensemble from.
        :type args: DataFrames
        :return: Transformed dataframe representing the ensemble.
        :rtype: pd.DataFrame
        """
        assert_same_structure_list(args)
        n, m = args[0].shape
        # number of models in the ensemble
        n_labels = len(args)
        data = np.zeros(n * m * n_labels).reshape(n, m, n_labels)
        for i, d in enumerate(args):
            data[:, :, i] = d
        labels = np.sum(data, axis=2) >= self.n_min
        data_mixed = pd.DataFrame(labels, columns=args[0].columns, index=args[0].index)
        return data_mixed


class ParquetWriter:
    """
    A class to write pandas DataFrame to Parquet format.

    Parameters:
    :param format: The format of the data. "long" for long format and
    "wide" for wide format.
    :param path: The path to the output Parquet file.
    :param value_name: The column name for the values in long format.
    Default is "value".
    """

    def __init__(self, format: str, path: Union[Path, str], value_name: str = "value"):
        """
        Initialize the ParquetWriter object.

        :param format: The format of the data. "long" for long format and "wide"
        for wide format.
        :param path: The path to the output Parquet file.
        :param value_name: The column name for the values in long format.
        Default is "value".
        """
        self.format = format
        self.path = path
        self.value_name = value_name

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Write the given DataFrame to a Parquet file in the specified format.

        :param data: The DataFrame to be written to Parquet.
        :return: The same DataFrame that was input, unchanged.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        if self.format == "long":
            data_long = to_long(data, self.value_name)
            data_long.to_parquet(self.path)
        else:
            data.to_parquet(self.path)
        return data


class SummaryWriter(Transformation):
    """
    A class for generating a summary DataFrame and writing it to a Parquet file.

    Usage Example:
        writer = SummaryWriter('/path/to/summary.parquet')
        transformed_data = writer.transform(data)
    """

    def __init__(self, path: str):
        """
        Initialize a SummaryWriter instance.

        :param path:
            The file path to save the Parquet file.
        """
        self.path = path

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary DataFrame and write it to a Parquet file.

        :param data:
            The input Arrow Table data.

        :return:
            The unmodified input data.
        """
        summary_df = summary(data)  # Replace with your actual summary function call
        summary_df.to_parquet(self.path)
        return data
