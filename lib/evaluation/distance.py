from sklearn.metrics import pairwise_distances


def calculate_distance(test_embeddings, gallery_embeddings, metric='cosine'):
    distances = pairwise_distances(test_embeddings, gallery_embeddings, metric)

    return distances