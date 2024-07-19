import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from epylabel.utils import to_wide
matplotlib.use("qt5agg")


def compute_mean_and_median(df):

    df['block'] = (df['label'] != df['label'].shift()).cumsum()

    # Compute run lengths
    run_lengths = df.groupby(['location', 'label', 'block']).size().reset_index(name='run_length')

    # Compute average run length per location and label
    summary_run_length = run_lengths.groupby(['location', 'label'])['run_length'].agg(['mean', 'median']).reset_index()

    # drop label==0 periods
    summary_run_length = summary_run_length[summary_run_length['label']==1]

    #clean up dataframe
    summary_run_length = summary_run_length.set_index('location').drop(columns='label').rename(columns={'mean': 'mean_label_length', 'median': 'median_label_length'})
    return(summary_run_length)


def process_region(region, algs):
    summaries = {}
    labels = {}
    for alg in algs:
        summary = pd.read_parquet(f"output/{region}/summary/{alg}.parquet")
        lab = pd.read_parquet(f"output/{region}/labels/{alg}.parquet")
        summary_run_length = compute_mean_and_median(lab)
        summary_extended = pd.concat([summary, summary_run_length], axis=1)
        summary_extended["region"] = region
        summary_extended["alg"] = alg
        lab["region"] = region
        lab["alg"] = alg
        summaries[alg] = summary_extended
        labels[alg] = lab
    return pd.concat(summaries), pd.concat(labels)


def plot_all_algs_with_incidence(region, comb_labels, location, incidence_data, algs, alg_labels, fig_width=10, fig_height=6):
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    sns.lineplot(
        x='target',
        y='value',
        data=incidence_data[incidence_data["location"] == location],
        ax=ax1,
        color="black"
    )

    n_algs = len(algs)
    horizontal_segments = np.linspace(0, 1, n_algs + 1).tolist()
    ytick_positions = [x + horizontal_segments[1] / 2 for x in horizontal_segments]
    for i, alg in enumerate(algs):
        dat = comb_labels[(comb_labels["location"] == location) & (comb_labels["alg"] == alg)]
        colors = {True: 'firebrick', False: '#dae6f0'}
        for _, row in dat.iterrows():
            ax1.axvspan(row['target'], row['target'] + pd.Timedelta(1, unit='D'),
                        ymin=horizontal_segments[i],
                        ymax=horizontal_segments[i+1]-0.01,
                        facecolor=colors.get(row['label'], 'white'))

    #plt.title(f'Incidence and Signal Label for {region} {location}')
    plt.xlabel('')
    plt.ylabel('Incidence')
    ax2 = ax1.twinx()
    ax2.set_yticks(ytick_positions[0:n_algs])
    ax2.set_yticklabels(alg_labels)
    ax1.set_xlim(left=datetime(2019, 12, 15), right=datetime(2023, 11, 1))
    fig.tight_layout()
    return fig, ax1


def plot_bar_chart(summary_data):
    bar_chart = plt.figure(figsize=(10, 6))
    sns.barplot(x='n_labels', y='mean_label_length', data=summary_data, order=summary_data["n_labels"], errorbar = None)
    plt.title('n_labels vs mean_label_length')
    plt.xlabel('Number of Labels')
    plt.ylabel('Mean Label Length')
    return bar_chart


def plot_densities(summary_data, algs, regions, fill=False, legend_labels=None):
    fig, ax = plt.subplots(figsize=(3, 2))
    cols = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Greys', 'YlOrBr']
    simple_cols = ['blue', 'red', 'green', 'purple', 'orange', 'grey', 'brown']
    patches = []

    for idx, (alg, reg, col, simple_col) in enumerate(zip(algs, regions, cols, simple_cols)):
        sns.kdeplot(
            x=summary_data[reg][summary_data[reg]['alg'] == alg]['n_labels'],
            y=summary_data[reg][summary_data[reg]['alg'] == alg]['mean_label_length'],
            cmap=col,
            fill=fill,
            ax=ax,
            alpha=1.0,
            label=alg,
            #linewidth=0.5
            #log_scale=(10,5)
        )
        if legend_labels:
            label = legend_labels[idx]
            patches.append(mpatches.Patch(color=simple_col, label=label, alpha=0.5))

    plt.xlabel('Number of labels')
    plt.ylabel('Mean label length')
    if legend_labels:
        ax.legend(handles=patches)
    fig.tight_layout()
    return fig


# Function to convert week-year string to date
def week_year_to_date(week_year_str):
    if week_year_str is None or '/' not in week_year_str:
        return None

    week, year = week_year_str.split('/')

    try:
        week = int(week)
        year = int(year)
    except ValueError:
        return None

    # Calculate the first day of the year
    first_day_of_year = datetime(year, 1, 1)

    # Calculate the first Monday of the year
    day_of_week = first_day_of_year.weekday()
    if day_of_week <= 3:  # If it's Thursday or earlier
        first_monday = first_day_of_year - timedelta(days=day_of_week)
    else:  # If it's Friday or later
        first_monday = first_day_of_year + timedelta(days=(7 - day_of_week))

    # Calculate the date of the Monday of the given week
    date = first_monday + timedelta(weeks=week - 1)
    return date.strftime("%Y-%m-%d")


def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(matplotlib.colors.to_rgb(c1))
    c2 = np.array(matplotlib.colors.to_rgb(c2))
    return matplotlib.colors.to_hex((1 - mix) * c1 + mix * c2)


def plot_map_timeseries(dists, timeseries_binary, timeseries_original, dates, n_clusters=4):
    """
    Creates Figure 3: visualization of regional clusters and mean labels per cluster
    Plot spatial and temporal data along with clustering information.

    Parameters:
    -----------
    dists : array-like
        Precomputed distance matrix for clustering.
    timeseries_binary : pandas.DataFrame
        Binary time series data.
    timeseries_original : array-like
        Original time series data.
    dates : array-like
        Array of dates corresponding to the time series data.
    n_clusters : int, optional
        Number of clusters for Agglomerative Clustering. Default is 4.

    Returns:
    --------
    None

    Notes:
    ------
    This function plots spatial and temporal data along with clustering information. It uses the Agglomerative Clustering
    algorithm to cluster spatial locations based on a precomputed distance matrix. The resulting clusters are visualized
    on a map along with the time series data. Each cluster's mean incidence and label frequency over time are plotted.
    Additionally, marking symbols are drawn on the temporal plots to reference specific time windows.

    """
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="complete"
    )

    clustering = clustering.fit(dists)

    label_map = pd.DataFrame(
        {"location": timeseries_binary.columns.astype(int), "label": clustering.labels_.astype(str)}
    )
    timeseries_binary = timeseries_binary.values.T
    gdf = gpd.read_file("shape/SKLKBerlinBez.shp")
    gdf["location"] = gdf["LKID"]
    gdf = gdf.merge(label_map, on="location")

    mydpi = 96*2
    fig = plt.figure(figsize=(3885/mydpi, 1800/mydpi))  # specify size and dimension 3885 x 900
    ax1 = plt.subplot2grid((n_clusters, 5), (0, 0), colspan=2, rowspan=n_clusters)

    for i in range(n_clusters):
        plt.subplot2grid((n_clusters, 5), (i, 2))
        plt.subplot2grid((n_clusters, 5), (i, 3), colspan=2)

    gdf.plot(column="label", cmap="viridis", ax=ax1, legend=False)
    ax1.margins(0)

    for i, clabel in zip(range(1, len(fig.axes), 2), range(n_clusters)):
        ax1 = fig.axes[i]
        ax2 = fig.axes[i + 1]
        rows_clabel = np.where(clustering.labels_ == clabel)[0]
        # plot mean incidence per cluster
        cmap = plt.get_cmap('viridis', n_clusters)  # formerly cm.get_cmap
        # annote clusters
        ax1.annotate(f"Cluster {clabel + 1}",
                     xy=(0.7, 0.55),
                    xycoords="data",
                    size=26, ha='center',
                     )

        ax1.annotate(f"n = {len(rows_clabel)}",
                     xy=(0.7, 0.31),
                    xycoords="data",
                    size=22, ha='center',
                     )

        ax1.annotate(" ",
                          xy=(0.35, 0.45), xycoords="data",
                          ha="center", size=28,
                          bbox=dict(boxstyle="circle", facecolor=matplotlib.colors.rgb2hex(cmap(clabel)),
                                    edgecolor=matplotlib.colors.rgb2hex(cmap(clabel))),
                        )

        # plot mean label frequency per cluster using color gradient
        label_frequency = timeseries_binary[rows_clabel, :].mean(axis=0)
        for x in range(timeseries_binary.shape[1]):
            ax2.axvline(dates[x], color=colorFader("#ffffff", "#B22222", label_frequency[x]), linewidth=1)
        ax2.plot(dates, timeseries_original[rows_clabel, :].mean(axis=0),
                 label=clabel,
                 c="black", linewidth=2, )

        # draw marking symbols for referencing in text
        def draw_window_reference(ax, window_dates, window_name):
            # small arrow indicating time window (at each subplot)
            ax.annotate('', xy=(window_dates[0], -0.05),
                         xycoords=('data', 'axes fraction'),
                         xytext=(window_dates[1], -0.05),
                         textcoords=('data', 'axes fraction'),
                         arrowprops=dict(arrowstyle='|-|',
                                         color='darkblue',
                                         mutation_scale=4,
                                         lw=2.0,
                                         ls='-')
                         )
            label_posx = window_dates[0] + (window_dates[1] - window_dates[0])/2  # datetime + timedelta
            # corresponding text label
            ax.annotate(window_name, xy=(label_posx, -0.18),
                        xycoords=('data', 'axes fraction'),
                        xytext=(label_posx, -0.18),
                        textcoords=('data', 'axes fraction'),
                        ha="center", fontsize=17
                        )

        draw_window_reference(ax2, (np.datetime64("2020-09-20"), np.datetime64("2021-01-10")), "i")
        draw_window_reference(ax2, (np.datetime64("2021-10-01"), np.datetime64("2021-12-15")), "ii")
        draw_window_reference(ax2, (np.datetime64("2022-11-23"), np.datetime64("2023-03-01")), "iii")

    # remove axis ticks and plot borders
    for j, ax in enumerate(fig.axes):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        if not j == len(fig.axes) - 1:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
        else:
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='center', fontsize=18)

    plt.tight_layout()


if __name__ == "__main__":
    # Load Data
    all_algs = ["bcp", "sp", "wv", "bcp_sp", "bcp_wv", "sp_wv", "bcp_sp_wv"]
    base_algs_and_ensemble = ["bcp", "sp", "wv", "bcp_sp_wv"]
    alg_labels = ["BCP", "Shapelet", "Wave Finder", "Ensemble"]
    regions = ["DE", "BL", "LK"]

    combined_summaries = {}
    combined_labels = {}
    for region in regions:
        combined_summaries[region], combined_labels[region] = process_region(region, all_algs)

    incidence_LK = pd.read_parquet("output/LK/incidence.parquet")
    incidence_DE = pd.read_parquet("output/DE/incidence.parquet")
    incidence_BL = pd.read_parquet("output/BL/incidence.parquet")

    # Incidence plots with algo labels
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    fig_LK, ax_LK = plot_all_algs_with_incidence("LK", combined_labels["LK"], 1001, incidence_LK, base_algs_and_ensemble, alg_labels, fig_width=6.475716, fig_height=1.5)
    fig_LK.savefig('output/inc_all_algs_1001.png', dpi=600)

    fig_DE, ax_DE = plot_all_algs_with_incidence("DE", combined_labels["DE"], 0, incidence_DE, base_algs_and_ensemble, alg_labels, fig_width=6.475716, fig_height=1.5)
    fig_DE.savefig('output/inc_all_algs_0.png', dpi=600)

    fig_BL, _ = plot_all_algs_with_incidence("BL", combined_labels["BL"], 1, incidence_BL, base_algs_and_ensemble, alg_labels, fig_width=6.475716, fig_height=1.5)
    fig_BL.savefig('output/inc_all_algs_1.png', dpi=600)

    # Infographic Plots
    infographic_labels = combined_labels["LK"][(combined_labels["LK"]["target"] >= "2021-06-01") & (combined_labels["LK"]["target"] < "2022-01-01")]
    infographic_incidence = incidence_LK[(incidence_LK["target"] >= "2021-06-01") & (incidence_LK["target"] < "2022-01-01")]

    fig_incidence, _ = plt.subplots(figsize=(10, 6))
    sns.lineplot(
            x='target',
            y='value',
            data=infographic_incidence[infographic_incidence["location"] == 1001],
            color="black"
        )
    plt.title("Incidence", fontsize=30)
    fig_incidence.savefig("output/infographic_incidence.svg")

    fig_infographic_ensemble, _ = plot_all_algs_with_incidence("LK", infographic_labels, 1001, infographic_incidence, ["bcp_sp_wv"],["bcp_sp_wv"])
    plt.title("Ensemble", fontsize=30)
    fig_infographic_ensemble.savefig('output/infographic_ensemble.svg')

    fig_infographic_bcp, _ = plot_all_algs_with_incidence("LK", infographic_labels, 1001, infographic_incidence, ["bcp"],["bcp"])
    plt.title("Bayesian Change Point", fontsize=30)
    fig_infographic_bcp.savefig('output/infographic_bcp.svg')

    fig_infographic_sp, _ = plot_all_algs_with_incidence("LK", infographic_labels, 1001, infographic_incidence, ["sp"], ["sp"])
    plt.title("Shapelet", fontsize=30)
    fig_infographic_sp.savefig('output/infographic_sp.svg')

    fig_infographic_wv, _ = plot_all_algs_with_incidence("LK", infographic_labels, 1001, infographic_incidence, ["wv"], ["wv"])
    plt.title("Wave Finder", fontsize=30)
    fig_infographic_wv.savefig('output/infographic_wv.svg')

    # BL Bar Chart
    summary_BL = combined_summaries["BL"].copy(deep=True).sort_values(by="n_labels")
    summary_BL_ensemble = summary_BL[summary_BL["alg"] == "bcp_sp_wv"]

    bar = plot_bar_chart(summary_BL_ensemble)
    bar.savefig('output/bar_chart_BL_ensemble.png')

    # Kernel Density Estimates
    dens_BL_ensemble = plot_densities(combined_summaries, algs=["bcp_sp_wv"], regions=["BL"], fill=True, legend_labels=None)
    dens_BL_ensemble.savefig('output/density_BL_ensemble.png', dpi=600)

    dens_LK_BL_ensemble = plot_densities(combined_summaries, algs=["bcp_sp_wv"]*2, regions=["LK", "BL"], legend_labels=["Counties", "States"])
    dens_LK_BL_ensemble.savefig('output/density_BL_and_LK_ensemble.png', dpi=600)

    dens_LK_all_algs = plot_densities(combined_summaries, algs=base_algs_and_ensemble, regions=["LK"] * 7, legend_labels=alg_labels)
    dens_LK_all_algs.savefig('output/density_LK_all_algs.png', dpi=600)


    # Define the data, from https://www.rki.de/DE/Content/Infekt/EpidBull/Archiv/2022/38/Art_01.html
    rki_wave_data = {
        'Phase': [0, 1, 2, '2a', '2b', 3, 4, 5, 6, '6a', '6b', 7, '7a', '7b', 8],
        'Name': ["Auftreten sporadischer Fälle", "Erste COVID-19-Welle", "Sommerplateau 2020", "", "",
                 "Zweite COVID-19-Welle", "Dritte COVID-19-Welle (VOC Alpha)", "Sommerplateau 2021",
                 "Vierte COVID-19-Welle (VOC Delta)", "(VOC Delta: Sommer)", "(VOC Delta: Herbst/Winter)",
                 "Fünfte COVID-19-Welle (VOC Omikron BA.1/BA.2)", "(Omikron-Sublinie BA.1)", "(Omikron-Sublinie BA.2)",
                 "Sechste COVID-19-Welle (VOC Omikron BA.5)"],
        'Beginn (KW)': ["5/2020", "10/2020", "21/2020", "21/2020", "31/2020", "40/2020", "9/2021", "24/2021",
                        "31/2021", "31/2021", "40/2021", "52/2021", "52/2021", "9/2022", "22/2022"],
        'Ende (KW)': ["9/2020", "20/2020", "39/2020", "30/2020", "39/2020", "8/2021", "23/2021", "30/2021",
                      "51/2021", "39/2021", "51/2021", "21/2022", "8/2022", "21/2022", None]
    }


    rki_wave_definitions = pd.DataFrame(rki_wave_data)

    rki_wave_definitions["Anfangsdatum"] = rki_wave_definitions['Beginn (KW)'].apply(week_year_to_date)

    # Initial label assignment
    rki_wave_definitions["label"] = [1,1,0,0,0,1,1,0,1,1,1,1,1,1,1]

    # Select relevant columns and add a new entry
    wave_start_dates = rki_wave_definitions[["Anfangsdatum", "label"]]
    wave_start_dates.loc[len(wave_start_dates)] = ["2022-09-22", 2]

    # Convert Anfangsdatum to datetime
    wave_start_dates['Anfangsdatum'] = pd.to_datetime(wave_start_dates['Anfangsdatum'])

    # Create a new date range
    start_date, end_date = "2020-01-03", "2023-10-13"
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex the DataFrame
    date_range_data = wave_start_dates.drop_duplicates().set_index('Anfangsdatum').reindex(date_range).reset_index()
    date_range_data.rename(columns={'index': 'Anfangsdatum'}, inplace=True)

    # Set first value
    date_range_data.iloc[0,1] = 1

    # Make same structure as combined_labels
    date_range_data.rename(columns={"Anfangsdatum": "target"}, inplace=True)
    date_range_data["location"] = 0
    date_range_data["block"] = 1
    date_range_data["region"] = "DE"


    rki_with_wv = date_range_data.copy(deep=True)

    # 1. dataset, simple RKI definitions including the second half of a wave
    date_range_data["alg"] = "rki_simple"
    date_range_data['label'].ffill(inplace=True) #forward fill
    date_range_data["label"] = date_range_data["label"].replace({1: True, 0: False, 2:np.nan})

    #2. dataset, combine RKI definitions with peaks from wavefinder to exclude the waning half of a wave
    rki_with_wv["alg"] = "rki"
    peaks=combined_labels["DE"][(combined_labels["DE"]["alg"]=="wv") &(combined_labels["DE"]["label"]==False)].groupby("block").head(1)["target"]
    # remove the mini peak in the second wave and anything after the officially labelled period
    peaks = peaks[(peaks != "2020-11-13") & (peaks < "2022-09-22")]
    rki_with_wv.loc[rki_with_wv["target"].isin(peaks), "label"] = 0
    rki_with_wv['label'].ffill(inplace=True) #forward fill
    rki_with_wv["label"] = rki_with_wv["label"].replace({1: True, 0: False, 2:np.nan})

    #combine and plot
    all_algs_with_rki = ["bcp", "sp", "wv", "bcp_sp_wv", "rki"] # add "rki_simple", back in if you want it plotted
    alg_labels_with_RKI = ["BCP", "Shapelet", "Wave Finder", "Ensemble", "Official (RKI)"]
    combined_labels_with_rki = pd.concat([combined_labels["DE"], date_range_data, rki_with_wv])

    fig_DE_rki, ax_DE_rki = plot_all_algs_with_incidence("DE", combined_labels_with_rki, 0, incidence_DE, all_algs_with_rki,  alg_labels_with_RKI, fig_width=6.475716, fig_height=1.5)
    fig_DE_rki.savefig('output/inc_all_algs_with_rki_0.png', dpi=600)


    ### create figure 3
    # load data anew for use in figure 3
    labels_LK_ensemble = pd.read_parquet("output/LK/labels/bcp_sp_wv.parquet")
    labels_LK_ensemble = to_wide(labels_LK_ensemble, "label")

    incidence_LK = pd.read_parquet("output/LK/incidence.parquet")

    # calculate pairwise distances between time series
    dists = pairwise_distances(labels_LK_ensemble.values.T, metric="jaccard")
    # plot figure 3 using 4 clusters
    plot_map_timeseries(dists, labels_LK_ensemble, to_wide(incidence_LK).T.values, incidence_LK.target.unique(), 4)

    plt.savefig("output/figure3.png", format="png", dpi=600)
