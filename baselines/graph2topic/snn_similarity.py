import numpy as np
from sklearn.neighbors import kneighbors_graph

def snn_similarity(X, neighbor_num):
    """Perform Shared Nearest Neighbor (SNN) clustering algorithm clustering.
    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or array of shape (n_samples, n_samples)
    A feature array
    neighbor_num : int
    K number of neighbors to consider for shared nearest neighbor similarity
    """

    # for each data point, find their set of K nearest neighbors
    knn_graph = kneighbors_graph(X, n_neighbors=neighbor_num, include_self=False)
    neighbors = np.array([set(knn_graph[i].nonzero()[1]) for i in range(len(X))])

    # the distance matrix is computed as the complementary of the proportion of shared neighbors between each pair of data points
    snn_distance_similarity = np.asarray([[get_snn_similarity(neighbors[i], neighbors[j]) for j in range(len(neighbors))] for i in range(len(neighbors))])

    return snn_distance_similarity

def get_snn_similarity(x0, x1):
    """Calculate the shared-neighbor similarity of two sets of nearest neighbors, normalized by the maximum number of shared neighbors"""

    return len(x0.intersection(x1)) / len(x0)


# def get_snn_distance(x0, x1):
#     """Calculate the shared-neighbor distance of two sets of nearest neighbors, normalized by the maximum number of shared neighbors"""

#     return 1 - get_snn_similarity(x0, x1)
