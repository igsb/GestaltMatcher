import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


class Cohort(object):
    def __init__(self, embeddings, gallery_file, test_file, split=None, gallery_file_2=None):
        self.embeddings = embeddings
        self.split = split

        gdf = pd.read_csv(gallery_file)
        tdf = pd.read_csv(test_file)
        if 'split' in gdf.columns:
            df = gdf[gdf['split'] == split]
        else:
            df = gdf
        gallery_subject_ids = df.subject.values
        gallery_image_ids = df.image_id.values
        gallery_synd_ids = df.label.values

        if gallery_file_2:
            gdf = pd.read_csv(gallery_file_2)
            if 'split' in gdf.columns:
                df = gdf[gdf['split'] == split]
            else:
                df = gdf
            gallery_subject_ids = np.append(gallery_subject_ids, df.subject.values)
            gallery_image_ids = np.append(gallery_image_ids, df.image_id.values)
            gallery_synd_ids = np.append(gallery_synd_ids, df.label.values)

        if 'split' in tdf.columns:
            df = tdf[tdf['split'] == split]
        else:
            df = tdf
        test_subject_ids = df.subject.values
        test_image_ids = df.image_id.values
        test_synd_ids = df.label.values

        self.gallery_image_ids = gallery_image_ids
        self.gallery_subject_ids = gallery_subject_ids
        self.gallery_synd_ids = gallery_synd_ids
        self.gallery_embeddings = np.array([self.embeddings[image_id] for image_id in self.gallery_image_ids])

        self.test_image_ids = test_image_ids
        self.test_subject_ids = test_subject_ids
        self.test_synd_ids = test_synd_ids
        self.test_embeddings = np.array([self.embeddings[image_id] for image_id in self.test_image_ids])

    def calculate_distance(self):
        self.distances = pairwise_distances(self.test_embeddings, self.gallery_embeddings, 'cosine')

    def rank(self):
        # Store results per image
        image_results = {}
        match_ranks = np.array([])

        for image_idx in range(0, self.distances.shape[0]):
            # Get the distance of image
            distances = self.distances[image_idx]

            # Sort gallery by given distance
            sorted_distance_indices = np.argsort(distances)
            sorted_synd_ids = self.gallery_synd_ids[sorted_distance_indices]

            target_synd_id = self.test_synd_ids[image_idx]

            # Take unique syndrome ids and reserve the index
            unique_ids, indices = np.unique(sorted_synd_ids, return_index=True)

            # Sort unique syndrome ids by reserved index
            sorted_synd_ids = unique_ids[np.argsort(indices)]
            match_rank = np.where(sorted_synd_ids == target_synd_id)[0] + 1

            # Append the first match rank
            match_ranks = np.append(match_ranks, match_rank)
            image_results[self.test_image_ids[image_idx]] = match_rank

        self.image_results = image_results
        self.evaluate_syndrome(match_ranks)

        return image_results

    def evaluate_syndrome(self, first_match_ranks):
        synd_ranks = {}
        for synd_id, rank in zip(self.test_synd_ids, first_match_ranks):
            if synd_id not in synd_ranks:
                synd_ranks[synd_id] = [rank]
            else:
                synd_ranks[synd_id].append(rank)
        self.topk = []
        for i in [1, 5, 10, 30]:
            synds_acc = np.array([sum(np.array(acc) <= i)/len(acc)*100 for _, acc in synd_ranks.items()])
            self.topk.append(synds_acc.mean())


def main():
    # Load look-up table for softmax
    with open('lookup_table.txt', 'r') as f:
        line = f.readlines()[1]
        synd_list = line.split(', ')
        synd_list[0] = int(synd_list[0].replace('[', ''))
        synd_list[-1] = int(synd_list[-1].replace(']', ''))
        synd_list = np.array([int(i) for i in synd_list])

    # Load GMDB embeddings
    df = pd.read_csv('encodings.csv', sep=';')
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

    print('Load GMDB embeddings: {}'.format(len(embeddings)))

    # Laod casia embeddings
    df = pd.read_csv('healthy_encodings.csv', sep=';')
    image_ids = []
    healthy_embeddings = {}
    for index, row in df.iterrows():
        image_ids.append(row[0].split('_')[0])
        embedding = row['representations'].split(', ')
        embedding[0] = float(embedding[0].replace('[', ''))
        embedding[-1] = float(embedding[-1].replace(']', ''))
        healthy_embeddings[int(row[0].split('_')[0])] = embedding

    print('Load CASIA embeddings: {}'.format(len(embeddings)))

    data_path = os.path.join('..', 'data', 'GestaltMatcherDB')

    # Evaluate Frequent set with Frequent gallery

    # Load splits for softmax
    df = pd.read_csv(os.path.join(data_path, 'gmdb_frequent_test_images_v1.csv'))
    frequent_test_subject_ids = df.subject.values
    frequent_test_image_ids = df.image_id.values
    frequent_test_synd_ids = df.label.values

    frequent_test_softmax = np.array(
        [image_softmax_ranks[image_id] if image_id in image_softmax_ranks else 139 for image_id in
         frequent_test_image_ids])

    # rank softmax
    synd_softmax = {}
    for synd_id, softmax in zip(frequent_test_synd_ids, frequent_test_softmax):
        match_rank = np.where(softmax == synd_id)[0]
        if synd_id not in synd_softmax:
            synd_softmax[synd_id] = [match_rank]
        else:
            synd_softmax[synd_id].append(match_rank)

    gmdb_softmax_topk = []
    for i in [1, 5, 10, 30]:
        synds_acc = np.array([sum(np.array(acc) < i) / len(acc) * 100 for _, acc in synd_softmax.items()])
        gmdb_softmax_topk.append(synds_acc.mean())

    # Evaluate GestaltMatcher
    cohort = Cohort(embeddings,
                    os.path.join(data_path, 'gmdb_frequent_gallery_images_v1.csv'),
                    os.path.join(data_path, 'gmdb_frequent_test_images_v1.csv'))
    cohort.calculate_distance()
    image_results = cohort.rank()
    gmdb_frequent_topk = cohort.topk

    gallery_images = 0
    test_images = 0
    cohort = Cohort(healthy_embeddings,
                    os.path.join(data_path, 'gmdb_frequent_gallery_images_v1.csv'),
                    os.path.join(data_path, 'gmdb_frequent_test_images_v1.csv'))
    cohort.calculate_distance()
    image_results = cohort.rank()
    gallery_images += len(set(cohort.gallery_image_ids))
    test_images += len(set(cohort.test_image_ids))
    healthy_frequent_topk = cohort.topk

    num_gallery_images = gallery_images
    num_test_images = test_images

    print('==================================================================')
    print('Test set     |Model      |Gallery |Test  |Top-1|Top-5|Top-10|Top-30')
    print('GMDB-frequent|Softmax    |-       |{}   |{:.2f}|{:.2f}|{:.2f} |{:.2f} |'.format(num_test_images,
                                                                                            gmdb_softmax_topk[0],
                                                                                            gmdb_softmax_topk[1],
                                                                                            gmdb_softmax_topk[2],
                                                                                            gmdb_softmax_topk[3]))
    print('GMDB-frequent|Enc-GMDB   |{}    |{}   |{:.2f}|{:.2f}|{:.2f} |{:.2f} |'.format(num_gallery_images,
                                                                                         num_test_images,
                                                                                         gmdb_frequent_topk[0],
                                                                                         gmdb_frequent_topk[1],
                                                                                         gmdb_frequent_topk[2],
                                                                                         gmdb_frequent_topk[3]))
    print('GMDB-frequent|Enc-healthy|{}    |{}   |{:.2f}|{:.2f}|{:.2f} |{:.2f} |'.format(num_gallery_images,
                                                                                         num_test_images,
                                                                                         healthy_frequent_topk[0],
                                                                                         healthy_frequent_topk[1],
                                                                                         healthy_frequent_topk[2],
                                                                                         healthy_frequent_topk[3]))


    # Evaluate Rare set with Rare gallery

    topks = []
    for i in range(10):
        cohort = Cohort(embeddings,
                        os.path.join(data_path, 'gmdb_rare_gallery_images_v1.csv'),
                        os.path.join(data_path, 'gmdb_rare_test_images_v1.csv'), i)
        cohort.calculate_distance()
        image_results = cohort.rank()
        topks.append(cohort.topk)
    topks = np.array(topks)
    gmdb_rare_topk = topks.mean(axis=0)

    topks = []
    gallery_images = 0
    test_images = 0
    for i in range(10):
        cohort = Cohort(healthy_embeddings,
                        os.path.join(data_path, 'gmdb_rare_gallery_images_v1.csv'),
                        os.path.join(data_path, 'gmdb_rare_test_images_v1.csv'), i)
        cohort.calculate_distance()
        image_results = cohort.rank()
        topks.append(cohort.topk)
        gallery_images += len(set(cohort.gallery_image_ids))
        test_images += len(set(cohort.test_image_ids))
    topks = np.array(topks)
    healthy_rare_topk = topks.mean(axis=0)

    num_gallery_images = gallery_images / 10
    num_test_images = test_images / 10
    print('==================================================================')
    print('Test set     |Model      |Gallery |Test  |Top-1|Top-5|Top-10|Top-30')
    print('GMDB-rare    |Enc-GMDB   |{}   |{} |{:.2f}|{:.2f}|{:.2f} |{:.2f} |'.format(num_gallery_images, num_test_images,
                                                                                    gmdb_rare_topk[0], gmdb_rare_topk[1],
                                                                                    gmdb_rare_topk[2], gmdb_rare_topk[3]))
    print('GMDB-rare    |Enc-healthy|{}   |{} |{:.2f}|{:.2f}|{:.2f} |{:.2f} |'.format(num_gallery_images, num_test_images,
                                                                                    healthy_rare_topk[0], healthy_rare_topk[1],
                                                                                    healthy_rare_topk[2], healthy_rare_topk[3]))

    # Evaluate Frequent set with unified gallery (Frequent + Rare)
    cohort = Cohort(embeddings,
                    os.path.join(data_path, 'gmdb_frequent_gallery_images_v1.csv'),
                    os.path.join(data_path, 'gmdb_frequent_test_images_v1.csv'),
                    split=0,
                    gallery_file_2= os.path.join(data_path, 'gmdb_rare_gallery_images_v1.csv'))
    cohort.calculate_distance()
    image_results = cohort.rank()
    gmdb_frequent_topk = cohort.topk

    gallery_images = 0
    test_images = 0
    cohort = Cohort(healthy_embeddings,
                    os.path.join(data_path, 'gmdb_frequent_gallery_images_v1.csv'),
                    os.path.join(data_path, 'gmdb_frequent_test_images_v1.csv'),
                    split=0,
                    gallery_file_2=os.path.join(data_path, 'gmdb_rare_gallery_images_v1.csv'))
    cohort.calculate_distance()
    image_results = cohort.rank()
    gallery_images += len(set(cohort.gallery_image_ids))
    test_images += len(set(cohort.test_image_ids))
    healthy_frequent_topk = cohort.topk

    num_gallery_images = gallery_images
    num_test_images = test_images

    print('==================================================================')
    print('Test set     |Model      |Gallery |Test  |Top-1|Top-5|Top-10|Top-30')
    print('GMDB-frequent|Enc-GMDB   |{}    |{}   |{:.2f}|{:.2f}|{:.2f} |{:.2f} |'.format(num_gallery_images,
                                                                                  num_test_images,
                                                                                  gmdb_frequent_topk[0],
                                                                                  gmdb_frequent_topk[1],
                                                                                  gmdb_frequent_topk[2],
                                                                                  gmdb_frequent_topk[3]))
    print('GMDB-frequent|Enc-healthy|{}    |{}   |{:.2f}|{:.2f}|{:.2f} |{:.2f} |'.format(num_gallery_images,
                                                                                  num_test_images,
                                                                                  healthy_frequent_topk[0],
                                                                                  healthy_frequent_topk[1],
                                                                                  healthy_frequent_topk[2],
                                                                                  healthy_frequent_topk[3]))

    # Evaluate Rare set with unified gallery (Frequent + Rare)

    topks = []
    for i in range(10):
        cohort = Cohort(embeddings,
                        os.path.join(data_path, 'gmdb_rare_gallery_images_v1.csv'),
                        os.path.join(data_path, 'gmdb_rare_test_images_v1.csv'),
                        split = i,
                        gallery_file_2=os.path.join(data_path, 'gmdb_frequent_gallery_images_v1.csv'))
        cohort.calculate_distance()
        image_results = cohort.rank()
        topks.append(cohort.topk)
    topks = np.array(topks)
    gmdb_rare_topk = topks.mean(axis=0)

    topks = []
    gallery_images = 0
    test_images = 0
    for i in range(10):
        cohort = Cohort(healthy_embeddings,
                        os.path.join(data_path, 'gmdb_rare_gallery_images_v1.csv'),
                        os.path.join(data_path, 'gmdb_rare_test_images_v1.csv'),
                        split=i,
                        gallery_file_2=os.path.join(data_path, 'gmdb_frequent_gallery_images_v1.csv'))
        cohort.calculate_distance()
        image_results = cohort.rank()
        topks.append(cohort.topk)
        gallery_images += len(set(cohort.gallery_image_ids))
        test_images += len(set(cohort.test_image_ids))
    topks = np.array(topks)
    healthy_rare_topk = topks.mean(axis=0)

    num_gallery_images = gallery_images / 10
    num_test_images = test_images / 10
    print('==================================================================')
    print('Test set     |Model      |Gallery |Test  |Top-1|Top-5|Top-10|Top-30')
    print('GMDB-rare    |Enc-GMDB   |{}  |{} |{:.2f} |{:.2f}|{:.2f} |{:.2f} |'.format(num_gallery_images, num_test_images,
                                                                                    gmdb_rare_topk[0], gmdb_rare_topk[1],
                                                                                    gmdb_rare_topk[2], gmdb_rare_topk[3]))
    print('GMDB-rare    |Enc-healthy|{}  |{} |{:.2f} |{:.2f}|{:.2f} |{:.2f} |'.format(num_gallery_images, num_test_images,
                                                                                    healthy_rare_topk[0], healthy_rare_topk[1],
                                                                                    healthy_rare_topk[2], healthy_rare_topk[3]))
    print('==================================================================')

if __name__ == "__main__":
    main()