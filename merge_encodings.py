import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from lib.datasets.utils import load_synds_list, load_deep_gestalt_encodings

def parse_args():
    parser = argparse.ArgumentParser(description='GestaltMatcher analysis:')
    parser.add_argument('--target_metadata', type=str, required=False,
                        help='Input metadata file of target cohort.')
    parser.add_argument('--target_embs', type=str, required=True,
                        help='Output embedding file of target cohort.')
    parser.add_argument('--emb_path', type=str, required=True,
                        help='Input metadata file of target cohort.')

    return parser.parse_args()


class merge_embeddings(object):
    def __init__(self, args):
        if args.target_metadata:
            target_df = pd.read_csv(args.target_metadata, sep='\t')
            target_image_ids = target_df.image_id.values
        else:
            target_image_ids = ['.'.join(i.split('.')[:-1]) for i in os.listdir(args.emb_path)]

        # Create output dir
        output_dir = os.path.split(args.target_embs)[0]
        if not os.path.exists(output_dir):
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)


        first_file = True
        output_emb_file = open(args.target_embs, 'w')
        for image_id in target_image_ids:
            emb_file_path = os.path.join(args.emb_path, str(image_id)+'.csv')
            emb_file = open(emb_file_path, 'r')
            lines = emb_file.readlines()

            count = 0

            # Strips the newline character
            for line in lines:
                if count > 0 or first_file:
                    first_file = False
                    output_emb_file.writelines(line)

                count += 1
            emb_file.close()
        output_emb_file.close()


def main():
    # Training settings
    args = parse_args()

    # Add logger

    print("Start merge_embeddings!")
    analysis = merge_embeddings(args)
    print('Analysis Done!')


if __name__ == "__main__":
    main()