# Graph2Topic
Graph2Topic is a topic model based on PLMs and community detections. Our approach is able to get high quality topics without pre-specifying the number of topics. 
## Using
python main.py --topic_model g2t --word_select_method tfidf_idfi --graph_method greedy_modularity --pretrained_model princeton-nlp/unsup-simcse-bert-base-uncased --seed 30 --dataset bbc --num_topics 10 --dim_size 5
1. topic_model: topic model being used, g2t only
2. word_select_method: approach of selecting topic words

*tfidf×idfi
* 
  
