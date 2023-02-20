from baselines.topic_model import TopicModel
from baselines.graph2topic import Graph2Topic
import pandas as pd
import gensim.corpora as corpora
from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel
import numpy as np


class Graph2TopicTM(TopicModel):
    def __init__(self, dataset, topic_model, num_topics, dim_size, word_select_method, graph_method,embedding, seed,data_name):
        super().__init__(dataset, topic_model, num_topics)
        print(f'Initialize Graph2TopicTM with num_topics={num_topics}, embedding={embedding}')
        self.dim_size = dim_size
        self.word_select_method = word_select_method
        self.embedding = embedding
        self.seed = seed
        self.data_name =data_name
        self.graph_method = graph_method
        
        # make sentences and token_lists
        self.token_lists = self.dataset.get_corpus()
        self.sentences = [' '.join(text_list) for text_list in self.token_lists]
        
        embedding_model = TransformerDocumentEmbeddings(embedding)
        self.model = Graph2Topic(embedding_model=embedding_model,embedding=self.embedding,
                             nr_topics=num_topics, 
                             dim_size=self.dim_size, 
                             word_select_method=self.word_select_method,
                             graph_method=self.graph_method, 
                             seed=self.seed,data_name=self.data_name)
    
    
    def train(self):
        self.topics = self.model.fit_transform(self.sentences)
    
    
    def evaluate(self):
        td_score = self._calculate_topic_diversity()
        cv_score, npmi_score, umass_score = self._calculate_cv_npmi_umass(self.sentences, self.topics)
        w2vm_L2 = self._calculate_w2vm_L2(self.token_lists,self.topics)

        return td_score, cv_score, npmi_score, umass_score, w2vm_L2
    
    
    def get_topics(self):
        return self.model.get_topics()
    
    
    def _calculate_topic_diversity(self):
        topic_keywords = self.model.get_topics()

        bertopic_topics = []
        for k,v in topic_keywords.items():
            temp = []
            for tup in v:
                temp.append(tup[0])
            bertopic_topics.append(temp)  

        unique_words = set()
        for topic in bertopic_topics:
            unique_words = unique_words.union(set(topic[:10]))
        td = len(unique_words) / (10 * len(bertopic_topics))

        return td


    def _calculate_cv_npmi_umass(self, docs, topics): 

        doc = pd.DataFrame({"Document": docs,
                        "ID": range(len(docs)),
                        "Topic": topics})
        documents_per_topic = doc.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = self.model._preprocess_text(documents_per_topic.Document.values)

        vectorizer = self.model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        words = vectorizer.get_feature_names()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in self.model.get_topic(topic)] 
                    for topic in range(len(set(topics))-1)]

        coherence_model = CoherenceModel(topics=topic_words, 
                                      texts=tokens, 
                                      corpus=corpus,
                                      dictionary=dictionary, 
                                      coherence='c_v')
        cv_coherence = coherence_model.get_coherence()

        coherence_model_npmi = CoherenceModel(topics=topic_words, 
                                      texts=tokens, 
                                      corpus=corpus,
                                      dictionary=dictionary, 
                                      coherence='c_npmi')
        npmi_coherence = coherence_model_npmi.get_coherence()

        coherence_model_u_mass = CoherenceModel(topics=topic_words, 
                                texts=tokens, 
                                corpus=corpus,
                                dictionary=dictionary, 
                                coherence='u_mass')
        u_mass_coherence = coherence_model_u_mass.get_coherence()

        return cv_coherence, npmi_coherence,  u_mass_coherence

    def _calculate_w2vm_L2(self, docs, topics):
        from gensim.models import Word2Vec
        model = Word2Vec(docs, vector_size=100, window=5, min_count=1, workers=4)
        word_embedding = model.wv

        w2v_l2_results = []
        topic_words = [[words for words, _ in self.model.get_topic(topic)] 
            for topic in range(len(set(topics))-1)]
        for t_word in topic_words:
            w2v_l2_ = topic_w2v(t_word, word_embedding)
            w2v_l2_results.append(w2v_l2_)

        return np.mean(w2v_l2_results)


def topic_w2v(topic_words, word_embedding):
    import scipy.spatial.distance as sci_dist

    l2_distance = 0.0

    n_top = len(topic_words)

    t = float(n_top)
    t = t * (t - 1.0)

    for word_i_idx in range(n_top):
        for word_j_idx in range(word_i_idx + 1, n_top):
            try:
                word_i = word_embedding[topic_words[word_i_idx]]
                word_j = word_embedding[topic_words[word_j_idx]]
            except KeyError:
                continue

            l2_distance += (sci_dist.sqeuclidean(word_i, word_j))

    l2_distance = l2_distance / t

    return l2_distance