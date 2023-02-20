# Graph2Topic
Graph2Topic is a topic model based on PLMs and community detections. Our approach is able to get high quality topics without pre-specifying the number of topics. 
## Using

python main.py --topic_model g2t --word_select_method tfidf_idfi --graph_method greedy_modularity --pretrained_model princeton-nlp/unsup-simcse-bert-base-uncased --seed 30 --dataset bbc --num_topics 10 --dim_size 5

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
6. dataset: 20ng, m10, bbc, crr, beer, asap, nlpcc & nlpcc_c are provide
7. num_topics: number of topic would be showed
8. dim_size: dimensions would be reduced
  
