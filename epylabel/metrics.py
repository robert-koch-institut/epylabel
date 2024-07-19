"""Summary statistics for labelled data"""
import pandas as pd

from epylabel.utils import signal_ids, to_long


def summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics from input data.

    This function calculates various summary statistics related to labeled signals
    in the input data.

    :param data: A pandas DataFrame containing the input data.
    :return: A pandas DataFrame containing summary statistics.
    """
    data = signal_ids(data)
    data_long = to_long(data, "label")
    data_long["signal"] = data_long["label"] > 0
    summary_stats = {}
    # signal days by location
    summary_stats["n_label_periods"] = data_long.groupby("location").signal.sum()
    # proportion signal days by location
    summary_stats["prop_labels"] = data_long.groupby("location").signal.mean()
    # n_labels
    summary_stats["n_labels"] = data_long.groupby("location").label.max()
    # longest label/signal
    summary_stats["max_label_length"] = (
        data_long.query("label != 0")
        .groupby(["location", "label"])
        .signal.count()
        .reset_index()
        .groupby("location")
        .signal.max()
    )
    # shortest label/signal
    summary_stats["min_label_length"] = (
        data_long.query("label != 0")
        .groupby(["location", "label"])
        .signal.count()
        .reset_index()
        .groupby("location")
        .signal.min()
    )
    return pd.concat(summary_stats, axis=1)
