"""
Author: Philip Oedi
Hier hab ich eher rumgespielt, der code steht soweit aber und könnte dann nur
nochmal etwas formalisiert werden
To-Do's:
- Distance metriken ausprobieren (jaccard, hamming, edit/leveshtein
    wären meine Vorschläge)
- Algorithmen: hierarchisch mit complete linkage, war das einzige,
    was halbwegs funktioniert hat

- anzahl cluster bestimmen: eda und auswahlkriterien ausprobieren
- knotenplot:
    es steht rudimentärer code für den knotenplot vor
    es müsste aber nochmal geschaut werden, dass das gut aussieht
    ideen: pro LK nur die top 10 edges oder so?

"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import LineString
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import MDS, TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet, fcluster
import matplotlib.cm as cm

from epylabel.labeler import Shapelet
from epylabel.utils import to_wide


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    cluster_labels = kwargs.pop("clustering_labels", None)
    if max_d:
        cluster_labels = fcluster(args[0], t=max_d, criterion="distance")

    n_clusters = len(np.unique(cluster_labels))

    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    plt.figure()
    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title(f'Hierarchical Clustering Dendrogram (truncated)\n max_d={max_d}, n_clust={n_clusters}')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def calculate_cophenetic(timeseries, metric="jaccard", linkage_method="complete"):
    """ Calculate the cophenetic correlation coefficient using scipy functions"""
    condensed_dists = pdist(timeseries, metric=metric)
    z = linkage(condensed_dists, method=linkage_method, metric=metric)

    c, _ = cophenet(z, condensed_dists)
    return round(c, 3)


def plot_maps(timeseries, metric="jaccard", linkage="complete"):

    dists = pairwise_distances(timeseries, metric=metric)
    cophenetic_corr = calculate_cophenetic(timeseries, metric, linkage)

    cluster_counts = [2, 5, 7, 9, 11, 13, 15, 17, 19]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, nc in zip(axes.flatten(), cluster_counts):
        if linkage == "ward":
            metric = "euclidean"
            clustering = AgglomerativeClustering(
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=nc, metric="precomputed", linkage=linkage
            )

        clustering = clustering.fit(dists)

        label_map = pd.DataFrame(
            {"location": out.columns.astype(int), "label": clustering.labels_.astype(str)}
        )

        gdf = gpd.read_file("shape/SKLKBerlinBez.shp")
        gdf["location"] = gdf["LKID"]
        gdf = gdf.merge(label_map, on="location")
        gdf.plot(column="label", cmap="viridis", ax=ax, legend=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"clusters={nc}")
        fig.suptitle(f"metric: {metric}, linkage: {linkage}, cophenetic={cophenetic_corr}")
    plt.tight_layout()


def reduce_dimensionality_mds(distances, ndim=2, clustering=None):
    if clustering is None:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="jaccard", linkage="complete", distance_threshold=0
        )
        clustering = clustering.fit(distances)

    mds = MDS(n_components=ndim, dissimilarity='precomputed', random_state=0, normalized_stress="auto")
    dists_transformed = mds.fit_transform(distances)

    fig, ax = plt.subplots(1, 1)
    if ndim == 3:
        ax = fig.add_subplot(projection='3d')
    ax.scatter(dists_transformed[:, 0], dists_transformed[:, 1],
               dists_transformed[:, 2], c=clustering.labels_)


def reduce_dimensionality_tsne(distances, ndim=2, clustering=None):
    if clustering is None:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="jaccard", linkage="complete", distance_threshold=0
        )
        clustering = clustering.fit(distances)

    tsne = TSNE(n_components=ndim, metric="precomputed", random_state=0, init="random")
    dists_tsne = tsne.fit_transform(distances)

    fig, ax = plt.subplots(1, 1)
    if ndim == 3:
        ax = fig.add_subplot(projection='3d')
    ax.scatter(dists_tsne[:, 0], dists_tsne[:, 1],
               dists_tsne[:, 2], c=clustering.labels_)


def plot_stress_mds(distances):
    stress = []
    # Max value for n_components
    max_range = 21
    for dim in range(1, max_range):
        # Set up the MDS object
        mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=0, normalized_stress="auto")
        # Apply MDS
        pts = mds.fit_transform(distances)
        # Retrieve the stress value
        stress.append(mds.stress_)
    # Plot stress vs. n_components
    plt.figure()
    plt.plot(range(1, max_range), stress)
    plt.xticks(range(1, max_range, 2))
    plt.xlabel('n_components')
    plt.ylabel('stress')


def min_max_scaling(column):
    """0 to 1 scaling of a numpy array"""
    min_val = column.min()
    max_val = column.max()
    scaled_column = (column - min_val) / (max_val - min_val)
    return scaled_column


def knotenplot(germany, scale=1):
    """
    Creates a map of germany with lines between the centroid of
    the counties. thickness corresponding to distance
    """

    # Sample distance matrix (replace with your actual distance matrix)
    distance_matrix = dists

    # Calculate centroids
    germany["centroid"] = germany.geometry.centroid

    # Create edges
    edges = []
    for i in range(len(germany)):
        for j in range(i + 1, len(germany)):
            source = germany["centroid"][i]
            target = germany["centroid"][j]
            distance = distance_matrix[i][j]
            edges.append({"geometry": LineString([source, target]), "distance": distance})

    edges_gdf = gpd.GeoDataFrame(edges, geometry="geometry")

    # Apply min-max scaling to a specific column
    edges_gdf["similarity"] = 1 - min_max_scaling(edges_gdf["distance"])
    # Plot
    edges_gdf = edges_gdf.sort_values("similarity").tail(10000)

    fig, ax = plt.subplots(figsize=(10, 10))
    germany.plot(ax=ax, color="lightgrey", edgecolor="black")
    edges_gdf.plot(
        ax=ax,
        linewidth=np.log(edges_gdf["similarity"]) * scale,
        alpha=edges_gdf["similarity"],
        legend=True,
    )
    plt.show()


def plot_regional_focus(distances, inputs, focus_location=11012):
    clustering = AgglomerativeClustering(
        n_clusters=9, metric="precomputed", linkage="complete"
    )

    clustering = clustering.fit(distances)

    label_map = pd.DataFrame(
        {"location": inputs.columns.astype(int), "label": clustering.labels_.astype(str)}
    )

    dists_df = pd.DataFrame(
    distances, columns=inputs.columns.astype(int), index=inputs.columns.astype(int)
    )

    dists_focus = dists_df.loc[focus_location].reset_index()
    dists_focus.columns = ["location", "distance"]

    gdf = gpd.read_file("shape/SKLKBerlinBez.shp")
    gdf["location"] = gdf["LKID"]
    gdf = gdf.merge(label_map, on="location")
    gdf = gdf.merge(dists_focus, on="location")

    # plot distance to focus_location
    mdist = gdf["distance"].mean()
    gdf["distance"] = gdf["distance"].replace(0, mdist)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column="distance", cmap="viridis", ax=ax, legend=True)

    # Add title and labels (customize as needed)
    plt.title(f"Colored Shapes by distance to {focus_location}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    gdf2 = gdf.query(f"location != {focus_location}")
    plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
    plt.scatter(gdf2["pop_pop"], gdf2["distance"], label="Data Points", color="blue")
    plt.xlabel("population")
    plt.ylabel("distance")
    plt.title(f"Distances to LK 11012, population:{gdf.loc[gdf['location'] == focus_location, 'pop_pop'][0]}")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":

    out = pd.read_parquet("output/LK/labels/bcp_sp_wv.parquet")
    out = to_wide(out, "label")

    # calculate pairwise distances
    dists = pairwise_distances(out.values.T, metric="jaccard")
    condensed_dists = pdist(out.values.T, metric="jaccard")  # scipy functionality
    z = linkage(condensed_dists, method="complete", metric="jaccard")

    # Calculate the cophenetic correlation coefficient
    c, _ = cophenet(z, condensed_dists)

    # Print the cophenetic correlation coefficient
    print("Cophenetic Correlation Coefficient:", c)

    # Create the dendrogram with colored branches
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    dendrogram(z, ax=ax)  # full dendogram
    plt.tight_layout()

    # condensed dendrogram showing sensitivity to max_d and small distances between clusters
    fancy_dendrogram(
        z,
        truncate_mode='lastp',
        p=26,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,  # useful in small plots so annotations don't overlap
        max_d=0.63
    )

    fancy_dendrogram(
        z,
        truncate_mode='lastp',
        p=26,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,  # useful in small plots so annotations don't overlap
        max_d=0.64
    )

    # visualize hierarchical clustering via regional map
    plot_maps(out.values.T)
    plot_maps(out.values.T, "euclidean")
    plot_maps(out.values.T, "correlation")
    plot_maps(out.values.T, "cosine")
    # only complete linkage really makes sense
    #plot_maps(out.values.T, "jaccard", "single")
    #plot_maps(out.values.T, "jaccard", "average")
    #plot_maps(out.values.T, "jaccard", "ward")


    plt.tight_layout()
    # Show the plot
    plt.show()


    ## other plots
    # plot_regional_focus(dists, out)
    # knotenplot(gdf, 1)
