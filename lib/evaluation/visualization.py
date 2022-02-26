import os
import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import scipy.spatial as sp, scipy.cluster.hierarchy as hc

from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def get_colors(num_clusters):
    """
    Get colors for TSNE

    :param num_clusters: int, the number of clusters
    :return list of colors
    """
    # Default colors
    colors = ['darkcyan', 'coral', 'navy', 'darkred', 'sandybrown', 'pink',
              'forestgreen', 'limegreen', 'darkgreen', 'springgreen', 'turquoise',
              'lightseagreen', 'paleturquoise', 'darkcyan', 'darkslateblue']
    # If we have clusters more than default colors
    if num_clusters > len(colors):
        colors = cm.rainbow(np.linspace(0, 1, num_clusters))
    return colors


def plot_tsne(embs, names, labels, output_path, syndrome_name_dict=None,
              show_metadata=False, synd_colors=None, title=None,
              gallery_dot_size=260, test_dot_size=300, file_type='svg',
              marker_dict=None, perplexity=15, not_show=False):
    # Perform TSNE
    embeddeds = TSNE(n_components=2, random_state=0,
                     metric="cosine", perplexity=perplexity, square_distances=True).fit_transform(embs)
    unique_syndrome_ids = np.unique(labels)
    num_unique_syndrome = len(unique_syndrome_ids)

    # Get the color for each syndrome
    if synd_colors:
        colors = synd_colors
    else:
        colors = get_colors(num_unique_syndrome)
    fig = plt.figure(figsize=(36, 24))

    # ordering by syndrome name and show ordered syndrome in legend
    if syndrome_name_dict:
        synd_names = np.array([syndrome_name_dict[synd_id] for synd_id in unique_syndrome_ids])
    else:
        synd_names = np.array([synd_id for synd_id in unique_syndrome_ids])
    unique_syndrome_ids = unique_syndrome_ids[np.argsort(synd_names)]
    colors = np.array(colors)
    colors = colors[np.argsort(synd_names)]

    enable_testing_metadata = 0
    # Draw images in each syndrome into 2D figure
    for syndrome_id, color in zip(unique_syndrome_ids, colors):
        dot_size = gallery_dot_size
        if marker_dict and syndrome_id in marker_dict:
            marker_type = marker_dict[syndrome_id]
        else:
            marker_type = 'o'
        label = syndrome_name_dict[syndrome_id] if syndrome_name_dict != None else syndrome_id
        plt.scatter(embeddeds[labels == syndrome_id, 0],
                    embeddeds[labels == syndrome_id, 1],
                    c=[color],
                    marker=marker_type,
                    label=label, s=dot_size)

    # Only add subject id and syndrome name in small syndrome
    if show_metadata:
        for x, y, syndrome_id, image_id in zip(embeddeds[:, 0], embeddeds[:, 1], labels, names):
            # only show testing metadata
            plt.annotate(image_id,
                         (x, y),
                         size=24)

    if title == None:
        title = "tsne"
    plt.title(title, fontsize=32)
    plt.legend(loc='center left', prop={'size': 36}, bbox_to_anchor=(1, 0.5), markerscale=2.5)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    filename = os.path.join(output_path, title + ".{}".format(file_type))
    plt.savefig(filename, bbox_inches="tight")
    if not not_show:
        plt.show()
    plt.close()


def get_image_by_name(name, path='.', file_type='jpg'):
    """
    get image by filename
    """
    path = os.path.join(path, '{}_crop_square.{}'.format(name, file_type))
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return img


def plot_clustering_heatmap(dist_df, rank_df, cnames, family_labels, gene_name, output_path='.',
                            file_format='png', ann_size=20, tick_size=18,
                            rotation=0, title_size=36, label_size=36, fig_size=(16, 16),
                            threshold=0.65, match_rank=30, display_match_box=False,
                            map_dict=None, source_type='distance', input_crops_path=None,
                            row_cluster=True, col_cluster=True, file_suffix='', linkage_method='single',
                            display_number=True):
    """
    Draw the heatmap and clustering dendrogram for the pairwise comparison of selected cohort.
    We can show either the pairwise distance or the pairwise rank in the cell.

    """

    # init parameter
    if source_type == 'distance':
        fmt = '.2g'
        center = 0.62
        vmax = None
        df = dist_df
        title = '{} pairwise distance'.format(gene_name)
    else:
        fmt = 'd'
        center = None
        vmax = 100
        df = rank_df
        title = '{} pairwise rank'.format(gene_name)

    # calculate cross fam matches
    diff_matrix = np.array([(i != family_labels) for i in family_labels])

    # plot clustering heatmap
    fig = plt.figure(figsize=fig_size)
    linkage = hc.linkage(sp.distance.squareform(dist_df), method=linkage_method)
    cg = sns.clustermap(df, figsize=fig_size, annot=display_number, fmt=fmt, center=center,
                        annot_kws={'size': ann_size}, cmap="Blues_r", vmax=vmax,
                        cbar_kws={'extend': 'max'}, yticklabels=cnames, xticklabels=cnames,
                        row_cluster=row_cluster, col_cluster=row_cluster,
                        row_linkage=linkage, col_linkage=linkage)
    plt.suptitle(title, fontsize=title_size, y=1.02)
    ax = cg.ax_heatmap
    ax.figure.axes[-1].tick_params(labelsize=20)

    # plot rectangular
    labels = [i.get_text() for i in ax.get_ymajorticklabels()]

    dist_matrix = dist_df
    rank_matrix = rank_df
    sorted_diff_matrix = pd.DataFrame(diff_matrix)
    sorted_diff_matrix.index = dist_matrix.index
    sorted_diff_matrix.columns = dist_matrix.columns
    match_matrix = (dist_matrix <= threshold) & sorted_diff_matrix & (rank_matrix <= match_rank)
    if display_match_box:
        for i in range(match_matrix.shape[0]):
            for j in range(match_matrix.shape[1]):
                if match_matrix.iloc[i, j] == 1:
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3))

    # plot patient's images
    if input_crops_path:
        ax_pos = ax.get_position()
        num_of_images = len(family_labels)
        if num_of_images < 4:
            x_offset = 1.14
            y_offset = -0.1
        elif num_of_images < 10:
            x_offset = 1.13
            y_offset = -0.08
        elif num_of_images >= 20:
            x_offset = 1.09
            y_offset = -0.07
        else:
            x_offset = 1.11
            y_offset = -0.07
        x_size = (ax_pos.y1 - ax_pos.y0) / num_of_images
        size = x_size * fig_size[0] * 0.8
        where = 'x'
        x_pos = np.linspace(0, 1, 2 * num_of_images + 1)[np.arange(1, 2 * num_of_images, 2)]
        y_pos = np.linspace(1, 0, 2 * num_of_images + 1)[np.arange(1, 2 * num_of_images, 2)]
        for which_axis in ['x', 'y']:
            for idx, label in enumerate(labels):
                if "x" in which_axis:
                    pos = x_pos[idx]
                    y = y_offset - x_size * 0.8
                    anchor, loc = (pos, y), 8
                else:
                    pos = y_pos[idx]
                    x = x_offset + x_size * 0.8
                    anchor, loc = (x, pos), 7
                _ax = inset_axes(
                    ax,
                    width=size,
                    height=size,
                    bbox_transform=ax.transAxes,
                    bbox_to_anchor=anchor,
                    loc=loc,
                )
                _ax.axison = False

                if map_dict and label in map_dict:
                    label = map_dict[label]
                if input_crops_path:
                    img = get_image_by_name(label, input_crops_path)
                    _ax.imshow(img, cmap='gray')
        ax.xaxis.set_label_coords(0.5, y - 0.01)
        ax.yaxis.set_label_coords(x + 0.01, 0.5)

    # set axis label
    ax.set_ylabel("Gallery images", fontsize=label_size)
    ax.set_xlabel("Test images", fontsize=label_size)
    plt.setp(ax.get_xticklabels(), rotation=rotation, fontsize=tick_size)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=tick_size)
    output_figure = os.path.join(output_path,
                                 '{}_validation_pairwise_{}{}.{}'.format(gene_name, source_type,
                                                                         file_suffix, file_format))
    plt.savefig(output_figure, format=file_format, bbox_inches='tight')