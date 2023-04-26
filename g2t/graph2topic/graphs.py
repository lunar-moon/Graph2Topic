import networkx as nx
import numpy as np
import pandas as pd

from .misc import *
from .misc import _save_pkl

def remove_edges(graph: nx.Graph, edge_weights: list, percentile_cutoff: int, remove_isolated_nodes: bool = 1) \
        -> nx.Graph:
    """
    remove_edges removes edges from graph that have a weight below the weight cutoff

    :param graph: doc embedding graph
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

    return graph

def create_semantic_graph(docs: list, doc_embeddings: list,
                          percentile_cutoff: int = 95, remove_isolated_nodes: bool = True,
                          method: str = "using_cutoff", top_n: int = 10):
    """
    create_semantic_graph creates a graph given the docs and their embeddings

    :param docs: list of docs which will be the nodes
    :param doc_embeddings: embeddings of the docs
    :param similarity_threshold: cosine similarity threshold value for the edges
    :param percentile_cutoff: percentile threshold value
    :param remove_isolated_nodes: boolean indicating if isolated nodes should be removed
    :param method: method used to trim the original graph
    :param top_n: is method is "using_top_n" then only the top_n highest weighted edges of each node is used

    :return:
        - doc embedding graph
    """
    assert len(docs) == len(doc_embeddings), "docs and doc_embeddings must have the same length"
    assert method in ["using_top_n", "using_cutoff"]

    edge_weights = []

    # create undirected graph
    graph = nx.Graph()

    # split doc embedding list in two halves
    first_half_length = int(len(doc_embeddings) / 2)
    first_half = doc_embeddings[:first_half_length]
    second_half = doc_embeddings[first_half_length:]
    
    # calculate cosine similarity
    sim_matrix = cosine_similarity(first_half, second_half) 
    similarity_threshold = pd.Series(sim_matrix.flatten()).mean()
    # print("similarity_threshold:"+str(similarity_threshold))

    for i in range(len(first_half)):
        i_sim_vector = sim_matrix[i]
        sim_i_sorted_index = sorted(range(len(i_sim_vector)), key=i_sim_vector.__getitem__, reverse=True)

        if method == "using_top_n":
            j_indices = sim_i_sorted_index[:top_n]
        else:
            j_indices = sim_i_sorted_index[:100]


        # iterate over all relevant adjacent nodes
        for j in j_indices:
            sim = i_sim_vector[j]

            if method == "using_cutoff" and sim < similarity_threshold:
                break

            else:
                doc_i = docs[i]
                doc_j = docs[first_half_length + j]
                graph.add_edge(doc_i, doc_j, weight=float(sim))
                edge_weights.append(sim)
    
    if method == "using_top_n":
        return remove_edges(graph, edge_weights, 50, remove_isolated_nodes)
    else:
        return remove_edges(graph, edge_weights, percentile_cutoff, remove_isolated_nodes)


# def sort_docs_by(graph, doc: str, doc_weights: dict):
#     """
#     sort_docs_by returns a tuple that is used to sort docs with each topic calculated from k-components

#     :param graph: doc embedding graph
#     :param doc: list of docs
#     :param doc_weights: list of doc weights

#     :return:
#         - doc_degree - degree of the node
#         - sim_score - pooled similarity score of node
#         - doc_weight - doc weight
#     """

#     neighbor_weights = []
#     for w_neighbor in graph.adj[doc]:
#         neighbor_weights.append(float(graph.adj[doc][w_neighbor]['weight']))

#     sim_score = np.average(neighbor_weights)
#     doc_degree = graph.degree(doc)
#     doc_weight = doc_weights[doc]

#     return doc_degree, sim_score, doc_weight
