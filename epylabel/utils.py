"""
Provides some functions to assert DataFrame structures and basic functionality, such
as transforming from wide to long and the other way around.
"""
import pandas as pd
from scipy.ndimage import label


def assert_all_bool(data: pd.DataFrame) -> None:
    """
    Asserts that all columns in the DataFrame have a boolean dtype.

    :param data: The DataFrame to be checked.
    :type data: pd.DataFrame

    :raises AssertionError: If any column's dtype is not bool.
    """
    assert data.dtypes.eq(bool).all()


def assert_same_structure_list(dataframes_list) -> None:
    """
    Asserts that all DataFrames in the list have the same column and index structure.

    :param dataframes_list: List of DataFrames to be checked.
    :type dataframes_list: args

    :raises AssertionError: If the DataFrames do not have the same structure.
    """
    if not all(isinstance(df, pd.DataFrame) for df in dataframes_list):
        raise ValueError("All elements in the list must be pandas DataFrames.")

    first_df = dataframes_list[0]
    for df in dataframes_list[1:]:
        if not df.columns.equals(first_df.columns) or not df.index.equals(
            first_df.index
        ):
            raise AssertionError(
                "DataFrames do not have the same column and index structure."
            )


def assert_same_structure(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """
    Asserts that two DataFrames have the same column and index structure.

    :param df1: The first DataFrame.
    :type df1: pd.DataFrame
    :param df2: The second DataFrame.
    :type df2: pd.DataFrame

    :raises AssertionError: If the DataFrames do not have the same structure.
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        raise ValueError("Both arguments must be pandas DataFrames.")

    if not df1.columns.equals(df2.columns) or not df1.index.equals(df2.index):
        raise AssertionError(
            "DataFrames do not have the same column and index structure."
        )


def to_long(data: pd.DataFrame, value_name) -> pd.DataFrame:
    """
    Converts a wide DataFrame to a long format, where the columns in the input data are
    locations, the rows targets (dates, weeks etc.) and the values are labels.

    :param data: The wide DataFrame to be converted.
    :type data: pd.DataFrame

    :returns: The long format DataFrame.
    :rtype: pd.DataFrame
    """
    return pd.melt(
        data.reset_index(), id_vars="target", var_name="location", value_name=value_name
    )


def to_wide(data: pd.DataFrame, col="value") -> pd.DataFrame:
    """
    Converts a long DataFrame to a wide format.

    :param data: The long DataFrame to be converted.
    :type data: pd.DataFrame

    :returns: The wide format DataFrame.
    :rtype: pd.DataFrame
    """
    pivot_df = data.pivot(index="target", columns="location", values=col)
    return pivot_df


def signal_ids(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies scipy.ndimage.label to each column in the DataFrame.

    :param data: The DataFrame to apply the labeling function to.
    :type data: pd.DataFrame

    :returns: The labeled DataFrame.
    :rtype: pd.DataFrame
    """
    return data.apply(lambda x: (label(x)[0]))
