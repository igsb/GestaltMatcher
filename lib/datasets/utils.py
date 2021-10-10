import numpy as np
import pandas as pd


def load_synds_list(filename='lookup_table.txt'):
    with open(filename, 'r') as f:
        line = f.readlines()[1]
        synd_list = line.split(', ')
        synd_list[0] = int(synd_list[0].replace('[', ''))
        synd_list[-1] = int(synd_list[-1].replace(']', ''))
        synd_list = np.array([int(i) for i in synd_list])
    return synd_list


def load_deep_gestalt_encodings(filename='encodings.csv', synd_list=[]):
    df = pd.read_csv(filename, sep=';')
    image_ids = []
    image_softmax_ranks = {}
    embeddings = {}
    for index, row in df.iterrows():
        image_ids.append(row[0].split('_')[0])
        embedding = row['representations'].split(', ')
        embedding[0] = float(embedding[0].replace('[', ''))
        embedding[-1] = float(embedding[-1].replace(']', ''))
        embeddings[int(row[0].split('_')[0])] = [float(i) for i in embedding]

        softmax = row['class_conf'].split(', ')
        softmax[0] = float(softmax[0].replace('[', ''))
        softmax[-1] = float(softmax[-1].replace(']', ''))
        softmax = np.array([float(i) for i in softmax])
        order = softmax.argsort()
        synd_ranks = synd_list[order[::-1]]
        image_softmax_ranks[int(row[0].split('_')[0])] = synd_ranks

    data = {'image_ids': np.array(image_ids),
            'embeddings': embeddings,
            'image_softmax_ranks': image_softmax_ranks}
    return data