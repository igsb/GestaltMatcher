import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


from lib.evaluation.distance import calculate_distance
from lib.evaluation.visualization import plot_clustering_heatmap, plot_tsne
from lib.datasets.utils import load_synds_list, load_deep_gestalt_encodings


def parse_args():
    parser = argparse.ArgumentParser(description='GestaltMatcher analysis:')
    parser.add_argument('--lookup_table', default='lookup_table.txt', type=str,
                        help='Mapping table of syndrome ID and the index of softmax.')
    parser.add_argument('--target_metadata', type=str, required=True,
                        help='Input metadata file of target cohort.')
    parser.add_argument('--target_embs', type=str, required=True,
                        help='Input embedding file of target cohort.')
    parser.add_argument('--gallery_embs', type=str, required=False,
                        help='Input embedding file of gallery.')
    parser.add_argument('--gallery_meta_data', type=str, required=False,
                        help='Input metadata file of gallery.')
    parser.add_argument('--output', type=str, required=True, default='.',
                        help='Output folder.')
    parser.add_argument('--linkage', default='average', type=str,
                        help='Linkage: single, average, complete, and ward. Default: average')
    parser.add_argument('--exp_name', default='', type=str,
                        help='Name of this analysis')
    parser.add_argument('--input_crops_path', default='', type=str,
                        help='Path to crops folder.')
    parser.add_argument('--output_image_type', default='png', type=str,
                        help='Output image type: png or svg')
    parser.add_argument('--gmdb_control_embs', default='dg_encodings_v1.csv', type=str,
                        help='gmdb embeddings used in pairwise rank matrix analysis.')

    return parser.parse_args()


class gestalt_matcher_analysis(object):
    def __init__(self, args):
        np.random.seed(1027)

        self.input_crops_path = args.input_crops_path
        self.linkage = args.linkage
        self.exp_name = args.exp_name
        self.output_image_type = args.output_image_type

        synd_list = load_synds_list(args.lookup_table)
        data = load_deep_gestalt_encodings(args.target_embs, synd_list)

        target_df = pd.read_csv(args.target_metadata, sep='\t')
        labels = target_df.label.values
        class_labels = target_df.class_label.values
        target_image_ids = target_df.image_id.values.astype(str)
        target_embeddings = np.array([data['embeddings'][i] for i in target_image_ids])

        # gmdb data used for control
        gmdb_data = load_deep_gestalt_encodings(args.gmdb_control_embs, synd_list)
        gmdb_image_ids = gmdb_data['image_ids']
        gmdb_image_ids = np.array([i for i in gmdb_image_ids if i not in target_image_ids])
        gmdb_embeddings = np.array([gmdb_data['embeddings'][i] for i in gmdb_image_ids])

        distance = calculate_distance(target_embeddings, target_embeddings)
        df = pd.DataFrame(distance)
        df.columns = target_image_ids
        df.index = target_image_ids

        # Create output dir
        self.output_dir = args.output
        if not os.path.exists(self.output_dir):
            path = Path(self.output_dir)
            path.mkdir(parents=True, exist_ok=True)

        # plot tSNE figure
        #all_labels = np.array(['GoF'] * len(target_image_ids) + ['LoF'] * len(control_image_ids))
        #synd_dict = {'GoF': 'GoF', 'LoF': 'LoF'}
        plot_tsne(target_embeddings, target_image_ids, class_labels, self.output_dir,
                  syndrome_name_dict=None, show_metadata=False, synd_colors=None,
                  title='{}_tSNE'.format(self.exp_name), gallery_dot_size=260, test_dot_size=300, file_type=self.output_image_type,
                  marker_dict=None, perplexity=15, not_show=True)

        # plot pairwise distance matrix
        self.plot_distance_matrix(df, labels, target_image_ids)

        # plot tSNE with control
        rand_index = np.random.randint(0, len(gmdb_embeddings), size=len(target_image_ids) * 10)
        selected_control_embeddings = np.array(gmdb_embeddings)[rand_index]
        selected_control_image_ids = np.array(gmdb_image_ids)[rand_index]

        all_image_ids = np.append(target_image_ids, selected_control_image_ids)
        all_labels = np.array(list(class_labels) + ['Control'] * len(selected_control_embeddings))
        
        plot_tsne(np.append(target_embeddings, selected_control_embeddings, axis=0), all_image_ids,
                  all_labels, self.output_dir,
                  syndrome_name_dict=None,
                  show_metadata=False, synd_colors=None, title='{}_tSNE_with_control'.format(self.exp_name),
                  gallery_dot_size=260, test_dot_size=300, file_type=self.output_image_type,
                  marker_dict=None, perplexity=15, not_show=True)

        # plot pairwise rank matrix
        cohort_image_ids = np.append(target_image_ids, gmdb_image_ids)
        cohort_embeddings = np.append(target_embeddings, gmdb_embeddings, axis=0)
        all_distances = calculate_distance(target_embeddings, cohort_embeddings)

        target_ranks = []
        for index, image_id in enumerate(target_image_ids):
            ranks = []
            # Get the distance of image
            distances = all_distances[index]

            # Sort gallery by given distance
            sorted_distance_indices = np.argsort(distances)
            sorted_image_ids = cohort_image_ids[sorted_distance_indices]
            for target_image_id in target_image_ids:
                ranks.append(np.where(sorted_image_ids == target_image_id)[0][0])
            target_ranks.append(ranks)

        target_ranks = np.array(target_ranks)
        target_ranks_df = pd.DataFrame(target_ranks, columns=target_image_ids, index=target_image_ids).T

        self.plot_rank_matrix(df, target_ranks_df, labels, target_image_ids)

    def plot_distance_matrix(self, df, labels, target_image_ids):
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
        map_dict = {str(i): str(j) for i, j in zip(labels, df.columns.values)}


        plot_clustering_heatmap(df, df, labels, target_image_ids, self.exp_name,
                                self.output_dir, self.output_image_type,
                                ann_size, tick_size, rotation, threshold=1.65, match_rank=30,
                                display_match_box=False,
                                source_type='distance', map_dict=map_dict,
                                input_crops_path=self.input_crops_path,
                                linkage_method=self.linkage,
                                row_cluster=True, col_cluster=True,
                                file_suffix='_{}'.format(self.linkage))


    def plot_rank_matrix(self, df, ranks_df, labels, target_image_ids):
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
        map_dict = {str(i): str(j) for i, j in zip(labels, df.columns.values)}

        plot_clustering_heatmap(df, ranks_df, labels, target_image_ids, self.exp_name,
                                self.output_dir, self.output_image_type,
                                ann_size, tick_size, rotation, threshold=1.65, match_rank=30,
                                display_match_box=False,
                                source_type='rank', map_dict=map_dict,
                                input_crops_path=self.input_crops_path,
                                linkage_method=self.linkage,
                                row_cluster=True, col_cluster=True, file_suffix='_{}'.format(self.linkage))


def main():
    # Training settings
    args = parse_args()

    # Add logger

    print("Start GestaltMatcher analysis!")
    analysis = gestalt_matcher_analysis(args)
    print('Analysis Done!')


if __name__ == "__main__":
    main()