from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import scipy.sparse as sp


class TFIDFi_IDFi(TfidfTransformer):

    def __init__(self, X_per_cluster, *args, **kwargs):
        print('====== Using TFIDF_IDFi ======')
        super().__init__(*args, **kwargs)
        self.X_per_cluster = X_per_cluster
        
    
    def socre(self):

        self._tfidfi = self.fit_transform(self.X_per_cluster)
        self._tfidfi = self._tfidfi.toarray()

        local_tfidf_transformer = TfidfTransformer()
        local_tfidf_transformer.fit_transform(self.X_per_cluster)
        self._idfi = local_tfidf_transformer.idf_
        
        scores = self._tfidfi * self._idfi
        #(50, 2949)
        #(2949,)
        # scores = normalize(scores, axis=1, norm='l1', copy=False)
        scores = sp.csr_matrix(scores)#(1, 50)

        return scores 

