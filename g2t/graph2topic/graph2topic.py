"""
Implemented based on CETopic 
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from .tfidf_idfi import TFIDF_IDFi
from .backend._utils import select_backend
import time

class Graph2Topic:

    def __init__(self, top_n_words=10, nr_topics=10, embedding_model=None, embedding=None,dim_size=-1, graph_method=None, seed=42):
        
        self.topics = None
        self.topic_sizes = None
        self.top_n_words = top_n_words
        self.nr_topics = nr_topics
        self.graph_method = graph_method
        self.embedding_model = embedding_model
        self.embedding = embedding.replace('/','-')
        self.vectorizer_model = CountVectorizer()
        
        self.dim_size = dim_size
        self.umap = None
        if self.dim_size != -1:
            self.umap = UMAP(n_neighbors=15, n_components=self.dim_size, min_dist=0.0, metric='cosine', random_state=seed)

        
    def fit_transform(self, documents, embeddings=None):
        start_time = time.time()
        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        print("---creating embedding---")
        if embeddings is None:
            self.embedding_model = select_backend(self.embedding_model)
            embeddings = self._extract_embeddings(documents.Document)
        else:
            if self.embedding_model is not None:
                self.embedding_model = select_backend(self.embedding_model)
        embedding_time = time.time()
        
        if self.umap is not None:
            embeddings = self._reduce_dimensionality(embeddings)
            umap_time = time.time()
        else:
            umap_time = embedding_time

        documents = self._detect_graph_topic(embeddings, documents)       
        
        graph_time = time.time()    
        # print(f' embedding_time:{round(embedding_time-start_time,4)} | umap_time:{round(umap_time-embedding_time,4)} | cluster_time:{round(graph_time-umap_time,4)}')
        
        self._extract_topics(documents)
        predictions = documents.Topic.to_list()

        return predictions


    def get_topics(self):
        return self.topics
    

    def get_topic(self, topic_id):
        if topic_id in self.topics:
            return self.topics[topic_id]
        else:
            return False


    def _extract_embeddings(self, documents):
        
        embeddings = self.embedding_model.embed_documents(documents,verbose=True)

        return embeddings
    

    def _reduce_dimensionality(self, embeddings):

        self.umap.fit(embeddings)
        reduced_embeddings = self.umap.transform(embeddings)
        
        return np.nan_to_num(reduced_embeddings)


    def _extract_topics(self, documents):
        
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        self.scores, words = self._weighting_words(documents_per_topic, documents)
        self.topics = self._extract_words_per_topic(words)


    def _weighting_words(self, documents_per_topic, all_documents):
        
        concatenated_documents = self._preprocess_text(documents_per_topic.Document.values)
        origin_documents = self._preprocess_text(all_documents.Document.values)
        
        # count the words in a cluster
        self.vectorizer_model.fit(concatenated_documents)
        words = self.vectorizer_model.get_feature_names_out()#list 

        # k * vocab k-num of topic
        X_per_cluster = self.vectorizer_model.transform(concatenated_documents)

        # D * vocab D-num of doc
        X_origin = self.vectorizer_model.transform(origin_documents)

        socres = TFIDF_IDFi(X_per_cluster, X_origin, all_documents).socre()

        return socres, words
    

    def _update_topic_size(self, documents):

        sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
        self.topic_sizes = dict(zip(sizes.Topic, sizes.Document))
        

    def _extract_words_per_topic(self, words):

        labels = sorted(list(self.topic_sizes.keys()))

        indices = self._top_n_idx_sparse(self.scores, 30)
        scores = self._top_n_values_sparse(self.scores, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        topics = {label: [(words[word_index], score)
                          if word_index and score > 0
                          else ("", 0.00001)
                          for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                          ]
                  for index, label in enumerate(labels)}

        topics = {label: values[:self.top_n_words] for label, values in topics.items()}

        return topics


    def _preprocess_text(self, documents):
        """ Basic preprocessing of text

        Steps:
            * Lower text
            * Replace \n and \t with whitespace
            * Only keep alpha-numerical characters
        """
        cleaned_documents = [doc.lower() for doc in documents]
        cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
        cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]

        return cleaned_documents
    

    @staticmethod
    def _top_n_idx_sparse(matrix, n):
        """ Return indices of top n values in each row of a sparse matrix

        Retrieved from:
            https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix

        Args:
            matrix: The sparse matrix from which to get the top n indices per row
            n: The number of highest values to extract from each row

        Returns:
            indices: The top n indices per row
        """
        indices = []
        for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            n_row_pick = min(n, ri - le)
            values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
            values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
            indices.append(values)
        return np.array(indices)
    

    @staticmethod
    def _top_n_values_sparse(matrix, indices):
        """ Return the top n values for each row in a sparse matrix

        Args:
            matrix: The sparse matrix from which to get the top n indices per row
            indices: The top n indices per row

        Returns:
            top_values: The top n scores per row
        """
        top_values = []
        for row, values in enumerate(indices):
            scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
            top_values.append(scores)
        return np.array(top_values)



    def _detect_graph_topic(self, embeddings, documents):
        from .graphs import create_semantic_graph

        # print("---detect graph topic---")
        graph = create_semantic_graph(documents["ID"], embeddings,percentile_cutoff=95)
        if self.graph_method =='k-components':
            from networkx.algorithms import approximation as approx
            components_all = approx.k_components(graph, min_density=0.8)# 0.8
            com = components_all[2]
        elif self.graph_method =='PLA':
            from networkx.algorithms.community import asyn_lpa_communities as lpa
            com = list(lpa(graph))
        elif self.graph_method =='greedy_modularity':
            from networkx.algorithms.community import greedy_modularity_communities
            com = greedy_modularity_communities(graph)
        elif self.graph_method =='louvain':
            import networkx.algorithms.community as nx_comm
            com = nx_comm.louvain_communities(graph,seed=123)
        elif self.graph_method == 'GN':#Girvan-Newman
            import networkx.algorithms.community.centrality as cen
            com = cen.girvan_newman(graph)
            com=list(sorted(c) for c in com)
            com = com[1]
        elif self.graph_method == 'LFM':
            from .overlap_community_detection import LFM
            com =  LFM(graph,0.7).execute()
            com=list(sorted(c) for c in com)
        elif self.graph_method == 'CPM':
            from networkx.algorithms.community import k_clique_communities
            com = k_clique_communities(graph, 2)
            com=list(sorted(c) for c in com)
        elif self.graph_method == 'COPRA':
            from .overlap_community_detection import COPRA
            com = COPRA(graph, 20, 3).execute()  
            com=list(sorted(c) for c in com)
        elif self.graph_method == 'SLPA':
            from .overlap_community_detection import SLPA
            com = SLPA(graph,20,0.5).execute()  
            com=list(sorted(c) for c in com)
                           
        com = sorted(com,key = lambda i:len(i),reverse=True)
        t=0
        for cluster_doc in com:
            if len(cluster_doc)>=5:
                if t >= self.nr_topics:
                    t+=1
                else:
                    for doc in cluster_doc:
                        # documents.Topic[documents.ID==doc] = t
                        documents.loc[documents.ID==doc, 'Topic'] = t
                    t+=1
        print("---Detect "+str(t)+" Topcis---")
        # documents.Topic[documents.Topic==None] = 999

        self._update_topic_size(documents)
        
        return documents
