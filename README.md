# Graph2Topic
**Graph to Topic**(*G2T*) is a topic model based on PLMs and community detections. Our approach is able to get high quality topics without pre-specifying the number of topics. 
![G2T](https://raw.githubusercontent.com/lunar-moon/Graph2Topic/v1.0/Images/Logo.png)
!!!Attention: there some bugs in program, I will fix it at soon as possible
## Prepare:
```
pip install graph2topictm
pip install 
```
## Use example:
```
from graph2topictm import graph2topictm
import time
from utils import *

dataset, sentences = prepare_dataset('bbc')
print(f'Using dataset: {dataset}, number of documents: {len(sentences)}')

token_lists = dataset.get_corpus()
sentences = [' '.join(text_list) for text_list in token_lists]
tm = graph2topictm.Graph2TopicTM(dataset=sentences, 
                embedding='princeton-nlp/unsup-simcse-bert-base-uncased')
start_time = time.time()
tm.train()
train_time = time.time()
print(f"Runtime of model:{round(train_time-start_time,4)}")
td_score, cv_score, npmi_score = tm.evaluate()
```

## Parameter of Class Graph2TopicTM:
1. topic_model: topic model being used, g2t only
2. word_select_method: approaches for selecting topic words
    1. tfidf_idfi(defalut)
    2. tfidf_tfi
    3. tfi
    4. tfidfi
3. graph_method: approaches for finding topic communities
    1. k-components
    2. PLA
    3. greedy_modularity(defalut)
    4. louvain
    5. LFM
    6. CPM
    7. COPRA
    8. SLPA
4. pretrained_model: PLMs for encoding texts, princeton-nlp/unsup-simcse-bert-base-uncased is recommended
5. seed: random seed
6. dataset: A list of documents, some datasets(20ng, m10, bbc, crr, beer, asap, nlpcc & nlpcc_c) are provided
7. num_topics: number of topic would be showed
8. dim_size: dimensions would be reduced
  
