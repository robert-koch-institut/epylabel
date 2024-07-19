import os
import tempfile

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from epylabel.labeler import (
    Bcp,
    Changerate,
    Ensemble,
    ExponentialGrowth,
    GapFiller,
    ParquetWriter,
    Shapelet,
    WaveFinder,
)
from epylabel.metrics import summary
from epylabel.pipeline import Pipeline
from epylabel.utils import (
    assert_all_bool,
    assert_same_structure,
    signal_ids,
    to_long,
    to_wide,
)


@pytest.fixture
def sample_data():
    # Sample data for the DataFram
    df = pd.read_parquet("test/test.parquet")
    locations = [1001, 1002, 1003]
    df = df.query("location in @locations")
    pivot_df = df.pivot(index="target", columns="location", values="value")
    return pivot_df


@pytest.fixture
def sample_cases():
    # Sample data for the DataFram
    return pd.read_parquet("test/test_cases.parquet")


@pytest.fixture
def sample_eg(sample_data):
    eg = ExponentialGrowth(n_days=14, thresh=0.8)
    labels = eg.transform(sample_data)
    return labels


@pytest.fixture
def sample_shapelet(sample_data):
    st = Shapelet(n_days=7, x_max=3, thresh=0.8)
    labels = st.transform(sample_data)
    return labels


@pytest.fixture
def sample_bcp(sample_cases):
    cr = Changerate(7)
    data = cr.transform(sample_cases)
    bcp = Bcp(d=1000, p0=0.001, thresh=1)
    labels = bcp.transform(data)
    return labels


def test_Changerate(sample_data):
    cr = Changerate(7)
    assert_same_structure(sample_data, cr.transform(sample_data))


def test_ExponentialGrowth(sample_data, sample_eg):
    assert_same_structure(sample_data, sample_eg)
    assert_all_bool(sample_eg)


def test_Shapelet(sample_data, sample_shapelet):
    assert_same_structure(sample_data, sample_shapelet)
    assert_all_bool(sample_shapelet)


def test_Bcp(sample_cases, sample_bcp):
    assert_same_structure(sample_cases, sample_bcp)
    assert_all_bool(sample_bcp)


def test_GapFiller(sample_eg):
    gf = GapFiller(n_days=7)
    labels = gf.transform(sample_eg)
    assert_same_structure(sample_eg, labels)
    assert_all_bool(labels)


def test_Ensemble(sample_eg, sample_shapelet):
    ens = Ensemble(n_min=2)
    labels = ens.transform(sample_eg, sample_shapelet)
    assert_same_structure(labels, sample_eg)
    assert_all_bool(labels)


def test_to_long(sample_eg):
    data = to_long(sample_eg, "label")
    # List of required columns
    required_columns = ["location", "target", "label"]

    # Assert that all required columns are present in the DataFrame
    assert all(
        col in data.columns for col in required_columns
    ), "The DataFrame is missing one or more required columns."


def test_signals_id(sample_eg):
    data = signal_ids(sample_eg)
    # Assert that all values are of type int
    assert data.dtypes.eq(int).all(), "Not all values in the DataFrame are of type int."
    # Assert that there are some values greater than 1
    assert (data > 1).any().any(), "No values greater than 1 found in the DataFrame."


def test_summary(sample_eg):
    report = summary(sample_eg)
    # List of columns to check for existence
    required_columns = [
        "n_label_periods",
        "prop_labels",
        "n_labels",
        "max_label_length",
        "min_label_length",
    ]

    # Assert that all required columns are in the DataFrame
    assert all(
        col in report.columns for col in required_columns
    ), "Not all required columns are present in the DataFrame."


def test_Pipeline(sample_data, sample_eg):
    eg = ExponentialGrowth(n_days=14, thresh=0.8)
    gf = GapFiller(n_days=7)
    pipe = Pipeline(steps=[eg, gf])
    out = pipe.transform(sample_data)
    out2 = gf.transform(sample_eg)
    assert_frame_equal(out, out2)
    eg2 = ExponentialGrowth(n_days=7, thresh=0.8)
    gf2 = GapFiller(n_days=7)
    pipe2 = Pipeline(steps=[eg2, gf2])
    out21 = pipe2.transform(sample_data)
    sample_eg2 = eg2.transform(sample_data)
    out22 = gf2.transform(sample_eg2)
    assert_frame_equal(out21, out22)
    en = Ensemble(n_min=2)
    pipe2 = Pipeline([en]).transform(out, out21)
    pipe2out = en.transform(out21, out)
    assert_frame_equal(pipe2, pipe2out)


def test_find_waves(sample_data):
    wv = WaveFinder(
        abs_prominence_threshold=5,  # minimum prominence
        rel_prominence_threshold=0.033,  # prominence relative to rel_to_constant
        rel_prominence_max_threshold=100,  # upper limit on relative prominence
        prominence_height_threshold=0.01,
        t_sep_a=35,
    )
    out = wv.find_waves(sample_data.iloc[:, 0])
    assert len(sample_data.iloc[:, 0]) == len(out)


def test_WaveFinder(sample_data):
    wv = WaveFinder(
        abs_prominence_threshold=5,  # minimum prominence
        rel_prominence_threshold=0.033,  # prominence relative to rel_to_constant
        rel_prominence_max_threshold=100,  # upper limit on relative prominence
        prominence_height_threshold=0.01,
        t_sep_a=35,
    )
    out = wv.transform(sample_data)
    assert_same_structure(sample_data, out)
    assert_all_bool(out)


def test_ParquetWriter(sample_data):
    temp_dir = tempfile.TemporaryDirectory()
    output_path = os.path.join(temp_dir.name, "long_output.parquet")
    writer = ParquetWriter("long", output_path)
    out = writer.transform(sample_data)
    assert_same_structure(sample_data, out)
    saved_df = pd.read_parquet(output_path)
    assert_same_structure(to_wide(saved_df), sample_data)
    assert_frame_equal(to_wide(saved_df), sample_data)
    temp_dir.cleanup()
