"""Script with pipeline to create the labels used for the paper"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from epylabel.labeler import (
    Bcp,
    Changerate,
    Ensemble,
    ParquetWriter,
    Shapelet,
    SummaryWriter,
    Transformation,
    WaveFinder,
)
from epylabel.pipeline import Pipeline
from epylabel.utils import to_wide


class StandardForm(Transformation):
    def __init__(self, col="Inzidenz_7-Tage"):
        """
        Specify column used as value
        :param col:
        """
        self.col = col

    def transform(self, data):
        """
        Makes the open data to the long format standard form
        :param data:
        :return:
        """
        if "Bundesland_id" in data.columns:
            data = data.rename(columns={"Bundesland_id": "location"})
        elif "Landkreis_id" in data.columns:
            data = data.rename(columns={"Landkreis_id": "location"})
        else:
            data["location"] = 0
        data = data.rename(columns={"Meldedatum": "target", self.col: "value"})
        if "Altersgruppe" in data.columns:
            data = data.query("Altersgruppe == '00+'")
        data = data[["target", "location", "value"]]
        data.sort_values(["location", "target"])
        data["target"] = pd.to_datetime(data["target"])
        return to_wide(data)


# bcp pipeline
# parameters
bcp_n_days = 7
bcp_d = 1000
bcp_p0 = 0.001
bcp_cr_ceiling = 6
# objects
cr = Changerate(n_days=bcp_n_days, changerate_ceiling=bcp_cr_ceiling)
bcp = Bcp(d=bcp_d, p0=bcp_p0, thresh=1)

# wave pipeline
wv_abs_prominence_threshold = 5  # minimum prominence
wv_prominence_height_threshold = (
    0.01  # prominence must be above a percentage of the peak height
)
wv_t_sep_a = 35

wv = WaveFinder(
    abs_prominence_threshold=wv_abs_prominence_threshold,
    prominence_height_threshold=wv_prominence_height_threshold,
    t_sep_a=wv_t_sep_a,
)

# shapelet pipeline
sp_n_days = 7
sp_thresh = 0.8
sp_x_max = 3
sp = Shapelet(n_days=sp_n_days, x_max=sp_x_max, thresh=sp_thresh)

# ensemble
ens = Ensemble(n_min=2)


data_map = {
    "LK": {
        "url": "https://raw.githubusercontent.com/robert-koch-institut/"
        "COVID-19_7-Tage-Inzidenz_in_Deutschland/main/"
        "COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv"
    },
    "BL": {
        "url": "https://raw.githubusercontent.com/robert-koch-institut/"
        "COVID-19_7-Tage-Inzidenz_in_Deutschland/main/"
        "COVID-19-Faelle_7-Tage-Inzidenz_Bundeslaender.csv"
    },
    "DE": {
        "url": "https://raw.githubusercontent.com/robert-koch-institut/"
        "COVID-19_7-Tage-Inzidenz_in_Deutschland/main/"
        "COVID-19-Faelle_7-Tage-Inzidenz_Deutschland.csv"
    },
}

output_dir = "output"


def writer(name, labels_path, summary_path):
    """
    Creates a tuple of transformation that would
    first write the long format file of a dataframe of labels
    and would then calculate and write the summary of that file.

    This is meant to call as a starred expression within a pipeline
    eg. Pipeline([step1, *writer("hi.parquet")])

    :param name:
    :param labels_path:
    :param summary_path:
    :return:
    """
    pw = ParquetWriter("long", labels_path / name, "label")
    sw = SummaryWriter(summary_path / name)
    return pw, sw


for geo, data_dict in tqdm(data_map.items()):
    data_rki = pd.read_csv(data_dict["url"])
    geo_path = Path(output_dir, geo)
    data_wide = Pipeline(
        [StandardForm(), ParquetWriter("long", geo_path / "incidence.parquet", "value")]
    ).transform(data_rki)
    data_wide_faelle = Pipeline(
        [
            StandardForm("Faelle_neu"),
            ParquetWriter("long", geo_path / "cases.parquet", "value"),
        ]
    ).transform(data_rki)

    labels_path = geo_path / "labels"
    summary_path = geo_path / "summary"

    def write(x):
        """
        partial function for writer. Predefines the labels
        and summary paths
        :param x:
        :return:
        """
        return writer(x, labels_path, summary_path)

    print(f"bcp labels for {geo}")
    bcp_labels = Pipeline([cr, bcp, *write("bcp.parquet")]).transform(data_wide_faelle)
    print(f"sp labels for {geo}")
    sp_labels = Pipeline([sp, *write("sp.parquet")]).transform(data_wide)
    print(f"wv labels for {geo}")
    wv_labels = Pipeline([wv, *write("wv.parquet")]).transform(data_wide)
    print(f"bcp_sp labels for {geo}")
    bcp_sp_labels = Pipeline([ens, *write("bcp_sp.parquet")]).transform(
        bcp_labels, sp_labels
    )
    print(f"bcp_wv labels for {geo}")
    bcp_wv_labels = Pipeline([ens, *write("bcp_wv.parquet")]).transform(
        bcp_labels, wv_labels
    )
    print(f"sp_wv labels for {geo}")
    sp_wv_labels = Pipeline([ens, *write("sp_wv.parquet")]).transform(
        sp_labels, wv_labels
    )
    print(f"bcp_sp_wv labels for {geo}")
    bcp_sp_wv_labels = Pipeline([ens, *write("bcp_sp_wv.parquet")]).transform(
        bcp_labels, sp_labels, wv_labels
    )
