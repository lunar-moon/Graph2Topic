import networkx as nx
import numpy as np
import pandas as pd

from .misc import *
import time
from .misc import _save_pkl
from .snn_similarity import snn_similarity

# from src.plot import make_plot

# plt.rcParams['figure.figsize'] = [16, 9]


def remove_edges(graph: nx.Graph, edge_weights: list, percentile_cutoff: int, remove_isolated_nodes: bool = 1) \
        -> nx.Graph:
    """
    remove_edges removes edges from graph that have a weight below the weight cutoff

    :param graph: word embedding graph
    :param edge_weights: list or edge weights
    :param percentile_cutoff: cutoff weight percentile
    :param remove_isolated_nodes: if 1, remove isolated nodes (default: 1)

    :return: processed graph
    """
    # remove edges that do not have a high enough similarity score
    min_cutoff_value = np.percentile(edge_weights, percentile_cutoff)

    graph_edge_weights = nx.get_edge_attributes(graph, "weight")

    edges_to_kill = []
    for edge in graph.edges():
        edge_weight = graph_edge_weights[edge]

        if edge_weight < min_cutoff_value:
            edges_to_kill.append(edge)

    for edge in edges_to_kill:
        graph.remove_edge(edge[0], edge[1])

    if remove_isolated_nodes:
        graph.remove_nodes_from(list(nx.isolates(graph)))

    print("剪枝后图的节点：",len(graph.nodes(data=True)))

    return graph


def create_networkx_graph_2(words: list, word_embeddings: list, similarity_threshold: float = 0.4,
                          percentile_cutoff: int = 70, remove_isolated_nodes: bool = True,
                          method: str = "using_cutoff", top_n: int = 10) :
    """
    create_networkx_graph creates a graph given the words and their embeddings

    :param words: list of words which will be the nodes
    :param word_embeddings: embeddings of the words
    :param similarity_threshold: cosine similarity threshold value for the edges
    :param percentile_cutoff: percentile threshold value
    :param remove_isolated_nodes: boolean indicating if isolated nodes should be removed
    :param method: method used to trim the original graph
    :param top_n: is method is "using_top_n" then only the top_n highest weighted edges of each node is used

    :return:
        - word embedding graph
        - graph creation time
    """
    assert len(words) == len(word_embeddings), "words and word_embeddings must have the same length"
    assert method in ["using_top_n", "using_cutoff"]
    number_of_embeddings = len(word_embeddings)
    if method == "using_cutoff":
        assert top_n < number_of_embeddings

    temp_sim_matrix = cosine_similarity(word_embeddings, word_embeddings)
    sim_matrix = [[sim if similarity_threshold < sim < 1 else np.NaN for sim in sim_vec]
                  for sim_vec in temp_sim_matrix]
    print()

    # create undirected graph
    graph = nx.Graph()

    start_time = time.process_time()

    similarity_sorted_indexes_matrix = [sorted(range(number_of_embeddings),
                                               key=sim_matrix[i].__getitem__,
                                               reverse=True)
                                        for i in range(number_of_embeddings)
                                        ]

    if method == "using_top_n":
        # get top n indexes for each node i
        j_indexes_matrix = [similarity_sorted_indexes_matrix[i][:top_n+1] for i in range(number_of_embeddings)]

    else:
        # method == "using_cutoff"

        # get percentile value for all edges
        # i_percentile_value = np.nanpercentile(sim_matrix, percentile_cutoff, axis=1)
        percentile_value = np.nanpercentile(sim_matrix, percentile_cutoff)

        j_indexes_matrix = [[j for j in range(number_of_embeddings)
                             if sim_matrix[i][j] > percentile_value and i != j]
                            for i in range(number_of_embeddings)]

    # create graph by adding edges and nodes
    for i in range(number_of_embeddings):
        for j in j_indexes_matrix[i]:

            if j == i:
                continue

            word_i = words[i]
            word_j = words[j]
            sim = sim_matrix[i][j]

            # check if the edge already exists
            if graph.has_edge(word_j, word_i):

                # the edge already exists
                old_weight = graph.get_edge_data(word_j, word_i, "weight")['weight']

                # remove edge if the old similarity score is lower than the new one
                if old_weight < sim:
                    graph.remove_edge(word_j, word_i)
                    # add new edge
                    graph.add_edge(word_i, word_j, weight=float(sim))

            else:
                # edge does not already exist -> add new edge
                graph.add_edge(word_i, word_j, weight=float(sim))

    graph_creation_time = time.process_time() - start_time

    return graph, graph_creation_time


# def create_networkx_graph(words: list, word_embeddings: list, similarity_threshold: float = 0.4,
#                           percentile_cutoff: int = 70, remove_isolated_nodes: bool = True,
#                           method: str = "using_cutoff", top_n: int = 10) -> Tuple[nx.Graph, float]
def create_networkx_graph(words: list, word_embeddings: list,
                          percentile_cutoff: int = 70, remove_isolated_nodes: bool = True,
                          method: str = "using_cutoff", top_n: int = 10):
    """
    create_networkx_graph creates a graph given the words and their embeddings

    :param words: list of words which will be the nodes
    :param word_embeddings: embeddings of the words
    :param similarity_threshold: cosine similarity threshold value for the edges
    :param percentile_cutoff: percentile threshold value
    :param remove_isolated_nodes: boolean indicating if isolated nodes should be removed
    :param method: method used to trim the original graph
    :param top_n: is method is "using_top_n" then only the top_n highest weighted edges of each node is used

    :return:
        - word embedding graph
        - graph creation time
    """
    assert len(words) == len(word_embeddings), "words and word_embeddings must have the same length"
    assert method in ["using_top_n", "using_cutoff"]

    start_time = time.process_time()
    edge_weights = []

    # create undirected graph
    graph = nx.Graph()

    # split word embedding list in two halves
    first_half_length = int(len(word_embeddings) / 2)
    first_half = word_embeddings[:first_half_length]
    second_half = word_embeddings[first_half_length:]
    
    #cos
    sim_matrix = cosine_similarity(first_half, second_half)
    # _save_pkl(sim_matrix.flatten(),"./cos/","20ng")
    # _save_pkl(sim_matrix,"./cos/","bbc")   

    # #snn_cos
    # sim_matrix = cosine_similarity(first_half, second_half)
    # sim_matrix = snn_similarity(sim_matrix,10)

    #norm_eu_dis_sim
    # sim_matrix = euclidean_distances(first_half, second_half)
    # # sc =StandardScaler()
    # # sim_matrix = sc.fit_transform(sim_matrix)
    # sim_matrix = 1 - sim_matrix
    # cos_analysis(sim_matrix.flatten())

    # #Pearson correlation coefficient
    # sim_matrix = np.corrcoef(first_half,second_half)
    # sim_matrix = np.maximum(sim_matrix)
    # cos_analysis(sim_matrix.flatten())    

    similarity_threshold = pd.Series(sim_matrix.flatten()).mean()
    print("similarity_threshold:"+str(similarity_threshold))

    print(len(words))

    for i in range(len(first_half)):
    # for i in range(len(words)):#for Pearson correlation coefficient
        # sort edges of node i by edge weight (similarity score)
        i_sim_vector = sim_matrix[i]
        sim_i_sorted_index = sorted(range(len(i_sim_vector)), key=i_sim_vector.__getitem__, reverse=True)

        if method == "using_top_n":
            j_indices = sim_i_sorted_index[:top_n]
        else:
            j_indices = sim_i_sorted_index[:100]
            # print(j_indices)

        # iterate over all relevant adjacent nodes
        for j in j_indices:
            sim = i_sim_vector[j]

            if method == "using_cutoff" and sim < similarity_threshold:
                break

            else:
                word_i = words[i]
                # print(first_half_length + j)
                word_j = words[first_half_length + j]
                # word_j=words[j]#for Pearson correlation coefficient
                graph.add_edge(word_i, word_j, weight=float(sim))
                edge_weights.append(sim)

    graph_creation_time = time.process_time() - start_time

    print("剪枝前节点的数量：",len(graph.nodes(data=True)))
    
    if method == "using_top_n":
        return remove_edges(graph, edge_weights, 50, remove_isolated_nodes), graph_creation_time
    else:
        return remove_edges(graph, edge_weights, percentile_cutoff, remove_isolated_nodes), graph_creation_time


def sort_words_by(graph, word: str, word_weights: dict):
    """
    sort_words_by returns a tuple that is used to sort words with each topic calculated from k-components

    :param graph: word embedding graph
    :param word: list of words
    :param word_weights: list of word weights

    :return:
        - w_degree - degree of the node
        - sim_score - pooled similarity score of node
        - w_weight - word weight
    """

    neighbor_weights = []
    for w_neighbor in graph.adj[word]:
        neighbor_weights.append(float(graph.adj[word][w_neighbor]['weight']))

    sim_score = np.average(neighbor_weights)
    w_degree = graph.degree(word)
    w_weight = word_weights[word]

    return w_degree, sim_score, w_weight
