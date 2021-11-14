import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


from lib.evaluation.distance import calculate_distance
from lib.evaluation.visualization import plot_clustering_heatmap
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


    return parser.parse_args()


class gestalt_matcher_analysis(object):
    def __init__(self, args):

        self.input_crops_path = args.input_crops_path
        self.linkage = args.linkage
        self.exp_name = args.exp_name

        synd_list = load_synds_list(args.lookup_table)
        data = load_deep_gestalt_encodings(args.target_embs, synd_list)

        target_df = pd.read_csv(args.target_metadata, sep='\t')
        labels = target_df.label.values
        target_image_ids = target_df.image_id.values
        target_embeddings = np.array([data['embeddings'][i] for i in target_image_ids])
        distance = calculate_distance(target_embeddings, target_embeddings)
        df = pd.DataFrame(distance)
        df.columns = target_image_ids
        df.index = target_image_ids

        # Create output dir
        self.output_dir = args.output
        if not os.path.exists(self.output_dir):
            path = Path(self.output_dir)
            path.mkdir(parents=True, exist_ok=True)

        # Just for the test purpose
        # Need to clean the code and move into different functions.

        IMAGE_TYPE = 'png'
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
                                self.output_dir, IMAGE_TYPE,
                                ann_size, tick_size, rotation, threshold=1.65, match_rank=30,
                                display_match_box=False,
                                source_type='distance', map_dict=map_dict,
                                input_crops_path=self.input_crops_path,
                                linkage_method=self.linkage,
                                row_cluster=True, col_cluster=True,
                                file_suffix='_{}'.format(self.linkage))


def main():
    # Training settings
    args = parse_args()

    # Add logger

    print("Start GestaltMatcher analysis!")
    analysis = gestalt_matcher_analysis(args)
    print('Analysis Done!')


if __name__ == "__main__":
    main()