import os
import json
import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn.manifold import TSNE
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from lib.evaluation.distance import calculate_distance
from lib.datasets.utils import load_synds_list, load_deep_gestalt_encodings
from lib.evaluation.visualization import plot_clustering_heatmap
from lib.evaluation.visualization import plot_tsne



def get_image_by_name(name, path='.', file_type='jpg'):
    """
    get image by filename
    """
    name = name.split('_')[0]
    file_path = os.path.join(path, '{}_crop_square.{}'.format(name, file_type))
    if not os.path.exists(file_path):
        file_path = os.path.join(path, '{}_rot_aligned.{}'.format(name, file_type))
    img = cv.imread(file_path, 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def plot_clustering_heatmap(dist_df, rank_df, cnames, family_labels, gene_name, output_path='.',
                            file_format='png', ann_size=20, tick_size=18,
                            rotation=0, title_size=36, label_size=36, fig_size=(16, 16),
                            threshold=0.65, match_rank=30, display_match_box=False,
                            map_dict=None, source_type='distance', input_crops_path=None,
                            row_cluster=True, col_cluster=True, file_suffix='', linkage_method='single',
                            display_number=True, default_x_offset=None, default_y_offset=None,
                            xaxis_ticks_x_offset=0.5, xaxis_ticks_y_offset=0.01, rotation_mode="anchor", ha="right"):
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
            x_offset = 1.16
            y_offset = -0.08
        elif num_of_images >= 20:
            x_offset = 1.09
            y_offset = -0.07
        else:
            x_offset = 1.11
            y_offset = -0.07
        if default_x_offset:
            x_offset = default_x_offset
        if default_y_offset:
            y_offset = default_y_offset
        x_size = (ax_pos.y1 - ax_pos.y0) / num_of_images
        size = x_size * fig_size[0] * 0.8
        where = 'x'
        x_pos = np.linspace(0, 1, 2 * num_of_images + 1)[np.arange(1, 2 * num_of_images, 2)]
        y_pos = np.linspace(1, 0, 2 * num_of_images + 1)[np.arange(1, 2 * num_of_images, 2)]
        for which_axis in ['x', 'y']:
            for idx, label in enumerate(labels):
                if "x" in which_axis:
                    pos = x_pos[idx]
                    y = y_offset - x_size * 1.1
                    anchor, loc = (pos, y), 8
                else:
                    pos = y_pos[idx]
                    x = x_offset + x_size * 1.4
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
                # label = label.split('_')[0]
                if map_dict and label in map_dict:
                    label = map_dict[label]
                if input_crops_path:
                    img = get_image_by_name(label, input_crops_path)
                    _ax.imshow(img, cmap='gray')
        ax.xaxis.set_label_coords(xaxis_ticks_x_offset, y - xaxis_ticks_y_offset)
        ax.yaxis.set_label_coords(x + 0.01, 0.5)

    # set axis label
    ax.set_ylabel("Gallery images", fontsize=label_size)
    ax.set_xlabel("Test images", fontsize=label_size)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_xticklabels(), rotation=rotation, fontsize=tick_size)  # , rotation_mode=rotation_mode, ha=ha)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=tick_size)
    output_figure = os.path.join(output_path,
                                 '{}_validation_pairwise_{}{}.{}'.format(gene_name, source_type,
                                                                         file_suffix, file_format))
    plt.savefig(output_figure, format=file_format, bbox_inches='tight')


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


def plot_tsne(dists, names, labels, output_path, syndrome_name_dict=None,
              show_metadata=False, synd_colors=None, title=None, show_title=True,
              gallery_dot_size=260, test_dot_size=300, file_type='svg',
              marker_dict=None, perplexity=15, not_show=False, x_offset=0, y_offset=0,
              legend_size=36, tsne_init='random', figure_size=(36, 24), metadata_font_size=24):
    # Perform TSNE
    embeddeds = TSNE(n_components=2, random_state=0, metric='precomputed', perplexity=perplexity, square_distances=True, init=tsne_init).fit_transform(dists)
    unique_syndrome_ids = np.unique(labels)
    num_unique_syndrome = len(unique_syndrome_ids)

    # Get the color for each syndrome
    if synd_colors:
        colors = synd_colors
    else:
        colors = get_colors(num_unique_syndrome)
    fig = plt.figure(figsize=figure_size)

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
                         (x+x_offset, y+y_offset),
                         size=metadata_font_size)

    if title == None:
        title = "tsne"
    if show_title:
        plt.title(title, fontsize=32)
    plt.legend(loc='center left', prop={'size': legend_size}, bbox_to_anchor=(1, 0.5), markerscale=2.5)
    plt.xticks(fontsize=56)
    plt.yticks(fontsize=56)

    filename = os.path.join(output_path, title + ".{}".format(file_type))
    plt.savefig(filename, bbox_inches="tight")
    if not not_show:
        plt.show()
    plt.close()

def main():

    gmdb_synd_to_dict = {}
    df = pd.read_csv('../data/GestaltMatcherDB/v1.0.3\gmdb_metadata\gmdb_frequent_gallery_images_v1.0.3.csv', sep=',')
    image_ids = df.image_id.values
    for _, row in df.iterrows():
        image_id = row['image_id']
        gmdb_synd_to_dict[image_id] = row['label']
    df = pd.read_csv('../data/GestaltMatcherDB/v1.0.3\gmdb_metadata\gmdb_frequent_test_images_v1.0.3.csv', sep=',')
    image_ids = np.append(image_ids, df.image_id.values)
    for _, row in df.iterrows():
        image_id = row['image_id']
        gmdb_synd_to_dict[image_id] = row['label']
    df = pd.read_csv('../data/GestaltMatcherDB/v1.0.3\gmdb_metadata\gmdb_rare_gallery_images_v1.0.3.csv', sep=',')
    image_ids = np.append(image_ids, df[df.split==0].image_id.values)
    for _, row in df.iterrows():
        image_id = row['image_id']
        gmdb_synd_to_dict[image_id] = row['label']
    df = pd.read_csv('../data/GestaltMatcherDB/v1.0.3\gmdb_metadata\gmdb_rare_test_images_v1.0.3.csv', sep=',')
    for _, row in df.iterrows():
        image_id = row['image_id']
        gmdb_synd_to_dict[image_id] = row['label']
    gmdb_release_image_ids = np.append(image_ids, df[df.split==0].image_id.values)


    mctt_df = pd.read_csv("../data/mctt/IDs_version_4_for_TC.txt", sep='\t')
    image_ids = []
    for _, row in mctt_df.iterrows():
        if row['ID v4'] in ['1260C09.2', 'M1273C15']:
            continue
        image_ids.append(row['ID v4'].replace('.', '-'))


    # Get all predictions
    representation_df = pd.read_csv("../GestaltMatcher-Arc/encodings_ensemble_v1.0.3_wo_pleiotropy_15012023.csv", delimiter=";")
    representation_df = representation_df.groupby('img_name').agg(lambda x: list(x)).reset_index()

    representation_df.representations = representation_df.representations.apply(lambda x: [json.loads(i) for i in x])
    representation_df.class_conf = representation_df.class_conf.apply(lambda x: [json.loads(i) for i in x])
    representation_df.img_name = representation_df.img_name.apply(lambda x: int(x.split('_')[0]))

    # Get target predictions
    target_representation_df = pd.read_csv("../GestaltMatcher-Arc/mctt_encodings_ensemble_v1.0.3_wo_pleiotropy_15012023.csv", delimiter=";")
    target_representation_df = target_representation_df.groupby('img_name').agg(lambda x: list(x)).reset_index()

    target_representation_df.representations = target_representation_df.representations.apply(lambda x: [json.loads(i) for i in x])
    target_representation_df.class_conf = target_representation_df.class_conf.apply(lambda x: [json.loads(i) for i in x])
    target_representation_df.img_name = target_representation_df.img_name.apply(lambda x: x.split('_')[0])

    gmdb_representation_df = representation_df[representation_df.img_name.isin(gmdb_release_image_ids)]
    gmdb_representation = gmdb_representation_df.representations.values

    target_representation = [target_representation_df[target_representation_df.img_name==i].representations.values[0] for i in image_ids]


    target_embs = [
        np.array([target_representation[j][i] for j in range(len(target_representation))]) for i in
        range(len(target_representation[0]))]
    gmdb_embs = [
        np.array([gmdb_representation[j][i] for j in range(len(gmdb_representation))]) for i in
        range(len(gmdb_representation[0]))]
    dists = np.stack([calculate_distance(target_embs[model_tta], target_embs[model_tta], 'cosine')
                      for model_tta in range(len(target_embs))], axis=1)

    name_mapping = {}
    subject_to_image = {}
    synd_dict = {}
    rename_image_ids = []

    for image_id in image_ids:
        name = image_id
        name_mapping[image_id] = name
        if 'C' in image_id:
            name = 'C' + image_id.split('C')[1]
        if 'N' in image_id:
            name = 'N' + image_id.split('N')[1]
        if name == 'C09-1':
            name = 'C09'
        subject_to_image[name] = str(image_id)
        synd_dict[name] = 'C-terminal' if 'C' in image_id else 'N-terminal'
        rename_image_ids.append(name)


    for i in range(12):
        gmdb_embs[i] = np.append(gmdb_embs[i], target_embs[i],axis=0)

    all_dists = np.stack([calculate_distance(target_embs[model_tta], gmdb_embs[model_tta], 'cosine')
                      for model_tta in range(0, 12)], axis=1)

    all_distance = np.mean(all_dists, axis=1)
    # sort_target_image_ids = image_ids
    sort_target_image_ids = rename_image_ids
    sort_cohort_image_ids = np.append(gmdb_representation_df.img_name.values.astype('str'), sort_target_image_ids)

    target_ranks = []
    for index, image_id in enumerate(sort_target_image_ids):
        ranks = []
        # Get the distance of image
        distances = all_distance[index]

        # Sort gallery by given distance
        sorted_distance_indices = np.argsort(distances)
        sorted_tmp_image_ids = sort_cohort_image_ids[sorted_distance_indices]
        for target_image_id in sort_target_image_ids:
            ranks.append(np.where(sorted_tmp_image_ids == target_image_id)[0][0])
        target_ranks.append(ranks)

    target_ranks = np.array(target_ranks)
    target_ranks_df = pd.DataFrame(target_ranks, columns=sort_target_image_ids, index=sort_target_image_ids).T

    sort_all_image_ids = np.array(sort_target_image_ids)


    OUTPUT_DIR = os.path.join('analysis_output', 'MCTT-output')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    distance = np.mean(dists, axis=1)
    df = pd.DataFrame(distance)
    df.columns = sort_all_image_ids
    df.index = sort_all_image_ids

    # Plot pairwise rank

    IMAGE_TYPE = 'svg'
    ann_size = 20
    tick_size = 18
    rotation = 0
    if len(df.columns.values) >= 40:
        ann_size = 8
        tick_size = 11
        rotation = 30
    elif len(df.columns.values) >= 20:
        ann_size = 8
        tick_size = 10
        rotation = 30
    elif len(df.columns.values) >= 15:
        ann_size = 16
        tick_size = 14
    cnames = df.columns.values
    #map_dict = {str(i): '6' for i in df.columns.values}
    type_cluster = 'single'
    type_cluster = 'complete'
    rotation = 'vertical'
    plot_clustering_heatmap(df, target_ranks_df, sort_all_image_ids, sort_all_image_ids, 'MN1',
                            OUTPUT_DIR, IMAGE_TYPE,
                            ann_size, tick_size, rotation, threshold=0.748, match_rank=300 ,#43,
                            display_match_box=False,
                            source_type='rank', map_dict=subject_to_image,
                            input_crops_path='../data/mctt/crops/',
                            linkage_method=type_cluster,
                            row_cluster=False, col_cluster=False, file_suffix='_{}'.format(type_cluster),
                            default_x_offset=1.05, default_y_offset=-0.06)

    # Plot pairwise distance

    IMAGE_TYPE = 'svg'
    ann_size = 20
    tick_size = 18
    rotation = 0
    if len(df.columns.values) >= 40:
        ann_size = 8
        tick_size = 8
        rotation = 30
    elif len(df.columns.values) >= 20:
        ann_size = 10
        tick_size = 10
        rotation = 30
    elif len(df.columns.values) >= 15:
        ann_size = 16
        tick_size = 14
    cnames = df.columns.values
    #map_dict = {str(i): '6' for i in df.columns.values}
    type_cluster = 'single'
    type_cluster = 'complete'
    rotation = 'vertical'
    plot_clustering_heatmap(df, df, sort_all_image_ids, sort_all_image_ids, 'MN1',
                            OUTPUT_DIR, IMAGE_TYPE,
                            ann_size, tick_size, rotation, threshold=0.748, match_rank=300 ,#43,
                            display_match_box=False,
                            source_type='distance', map_dict=subject_to_image,
                            input_crops_path='../data/mctt/crops/',
                            linkage_method=type_cluster,
                            row_cluster=True, col_cluster=True, file_suffix='_{}'.format(type_cluster),
                            default_x_offset=1.07, default_y_offset=-0.07)



    # Plot tSNE

    mn1_dist = np.mean(dists, axis=1)

    np.random.seed(15)
    synd_dict_2 = {i: i for i in synd_dict.values()}
    marker_dict = {i: 'o' if i == 'N-terminal' else 'X' for i in synd_dict_2}
    color_dict = ['tab:blue', 'tab:orange']
    all_labels = np.array([synd_dict[i] for i in rename_image_ids])
    t_image_ids = [i.split('_')[0] for i in rename_image_ids]
    synd_dict_2 = {i: i for i in synd_dict.values()}
    plot_tsne(mn1_dist, t_image_ids, all_labels, OUTPUT_DIR, syndrome_name_dict=synd_dict_2,
                  show_metadata=True, synd_colors=color_dict, title='tSNE_MCTT',
                  gallery_dot_size=3000, test_dot_size=400, file_type='svg', legend_size=100,
                  marker_dict=marker_dict, perplexity=30, x_offset=1.5, y_offset=2, show_title=False,
                  metadata_font_size=48, figure_size=(54, 48))


if __name__ == '__main__':
    main()