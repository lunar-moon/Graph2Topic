from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import numpy as np







def get_nearest_indices(embedding, list_embedding, n_nearest: int = 10) -> list:
    """
    get_most_similar_indices finds the indices that are most similar to the input embedding

    :param embedding: a single embedding
    :param list_embedding: a list of embeddings, the indices
    :param n_nearest: number of indices return

    :return: list of the nearest indices
    """

    sim_matrix = cosine_similarity(embedding.reshape(1, -1), list_embedding)[0]
    most_sim = np.argsort(sim_matrix, axis=None)[:: -1]

    return most_sim[:n_nearest]


import pickle

def _save_pkl(data,path,name):
    import pickle
    with open(str(path)+str(name) + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def _read_pkl(file_path):
    import pickle
    with open(file_path, 'rb') as f:           
        return pickle.load(f)

def cos_analysis(cos):
    from scipy.stats import skew,kurtosis
    print("==========")
    print("均值："+str(np.mean(cos)))
    print("标准差："+str(np.std(cos)))
    print("25/50/75分位数："+str(np.percentile(cos,25))+" / "+str(np.percentile(cos,50))+" / "+str(np.percentile(cos,75)))
    print("极差："+str(np.max(cos)-np.min(cos)))
    print("峰度系数："+str(kurtosis(cos)))
    print("偏度系数："+str(skew(cos)))

    no=0
    for i in cos:
        if i>0:
            no+=1
    print(no)

    print("==========")