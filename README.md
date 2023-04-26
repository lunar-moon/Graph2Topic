# Graph to Topic(G2T)
![G2T](https://github.com/lunar-moon/Graph2Topic/blob/v2.0/Images/logo.png)
**Graph to Topic**(*G2T*) is a topic model based on PLMs and community detections. Our approach is able to get high quality topics without pre-specifying the number of topics.
## Prepare:
```
#Not necessary, but conda is recommended
Conda create --n G2T python==3.9 
Conda activate G2T  

pip install graph2topictm -i https://pypi.org/simple
pip install -r requirements.txt 
```
## Use example:
```python
#get data
data = []#A list of documents
with open(r"./data/20NewsGroup/corpus.tsv","r",encoding='utf8')as f:
    for line in f.readlines():
        line = line.split('\t')[0]
        data.append(line)
f.close()

from g2t.graph2topictm import Graph2TopicTM
tm = Graph2TopicTM(dataset=data, 
                embedding='princeton-nlp/unsup-simcse-bert-base-uncased')
prediction = tm.train()
topics = tm.get_topics()
print(f'Topics: {topics}')

```

## Parameter of Class Graph2TopicTM:
1. graph_method: approaches for finding topic communities
    1. k-components
    2. PLA
    3. greedy_modularity(defalut)
    4. louvain
    5. LFM
    6. CPM
    7. COPRA
    8. SLPA
2. pretrained_model: PLMs for encoding texts, default is *bert-base-uncased*. We *sentence-transformers* library to get docmemnts representation, most PLMs in [huggingface](https://huggingface.co/models) could be used in g2t:
    1. *princeton-nlp/unsup-simcse-bert-base-uncased* is recommended for **English** data;
    2. *cyclone/simcse-chinese-roberta-wwm-ext* is recommened for **Chinese** data;
3. num_topics: number of topic would be showed, default is 10;
4. dim_size: dimensions would be reduced, default is 5;
## Acknowledgement
The code is implemented using [CETopic](https://github.com/hyintell/topicx)
