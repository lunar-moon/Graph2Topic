# Graph to Topic(G2T)
![logo](https://github.com/lunar-moon/Graph2Topic/blob/v2.0/Images/logo.png)
**Graph to Topic**(*G2T*) is a topic model based on PLMs and community detections. Our approach is able to get high quality topics without pre-specifying the number of topics. The main process of G2T is as follows:
![main process](https://github.com/lunar-moon/Graph2Topic/blob/v2.0/Images/模型图1-v2.png)
## Prepare:
```python
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
```
Example of output
Topics: {0: [('faith', 0.015225980397721653), ('religion', 0.013505623953838575), ('christian', 0.013041286150817297), ('doctrine', 0.012117303406529594), ('scripture', 0.011651357163244516), ('church', 0.009262452536868892), ('belief', 0.0078018569400895195), ('eternal', 0.007622325265905166), ('tradition', 0.007466917818556074), ('interpretation', 0.007049897251857179)], 1: [('file', 0.01598088366575847), ('server', 0.011975090077188815), ('window', 0.01115333082647189), ('widget', 0.010392458636425145), ('program', 0.007240145629352383), ('software', 0.007180846900590385), ('client', 0.006666002614300971), ('list', 0.006647050600026045), ('package', 0.006628054171955732), ('user', 0.006476425963713711)], 2: [('encryption', 0.036650112668314985), ('chip', 0.024879947870796094), ('clipper',0.024799643413291182), ('escrow', 0.022247509643059903), ('key', 0.019984459274609553), ('privacy', 0.017597544460276036), ('algorithm',0.012240709245025171), ('enforcement', 0.012019929086258466), ('government', 0.011955816470668994), ('security', 0.009639675079979204)], 3: [('space',0.02513160940465994), ('orbit', 0.024452557199861058), ('satellite', 0.018101063261669714), ('launch', 0.01772010383911628), ('shuttle',0.017072086969019122), ('solar', 0.01504678327703117), ('mission', 0.013618726017999095), ('rocket', 0.01173130020901071), ('station', 0.008293235928299387), ('flight',0.007494395721846273)], 4: [('scsi', 0.020447553937683136), ('drive', 0.01716853906498366), ('pin', 0.01332700675239151), ('ide', 0.013128479738754882), ('disk', 0.012284168431130586), ('chip', 0.010855934408040712), ('mhz', 0.010465707097541724), ('cable', 0.00967809652020599), ('controller', 0.009658752700338512), ('meg', 0.009490448499401926)], 5: [('armenian', 0.07277355469628521), ('turkish', 0.0518733094173384), ('genocide', 0.026576462535296767), ('greek', 0.024571758889747615), ('massacre', 0.01790756582596193), ('muslim', 0.016788498285298496), ('russian', 0.01653903839656462), ('village', 0.01426348700961069), ('population', 0.009790461263602277), ('government', 0.009153674718549476)], 6: [('window', 0.01720081713651007), ('disk', 0.010305136193377593), ('file', 0.010111290774951149), ('graphic',0.009972090377165446), ('software', 0.009350073860052575), ('color', 0.008573085022907989), ('frame', 0.00831184432839023), ('user', 0.008252393194885797), ('system', 0.007316502657577208), ('version', 0.0067474078535867356)], 7: [('patient', 0.030065548040147556), ('health', 0.02108286475394229), ('disease', 0.019342930639699656), ('insurance', 0.018201976825631294), ('doctor', 0.013319422159154742), ('medical', 0.01311321803118982), ('treatment', 0.011705307380905704), ('food',0.008312370712647413), ('drug', 0.008292675344308708), ('cell', 0.00827453017352833)], 8: [('atheist', 0.020001108950926088), ('belief', 0.01727369273452998), ('evidence', 0.014050043995273861), ('atheism', 0.014015115187002498), ('science', 0.012793607304810871), ('religion', 0.011324992703614072), ('existence', 0.010710375683394164), ('exist', 0.009366382011969265), ('question', 0.008486531687865792), ('faith', 0.008340472837289213)], 9: [('batf', 0.04172442706586647), ('fire', 0.016867852912078478), ('agent', 0.015509247036167758), ('gun', 0.013413381334369463), ('gas', 0.011767945814129061), ('warrant', 0.011380422210280308), ('weapon', 0.009583587364268516), ('police', 0.008202119833314925), ('tank', 0.00814800184637872), ('cult', 0.00789995071176192)]}
```
## Parameter of Class *Graph2TopicTM*:
1. graph_method: approaches for finding topic communities
    1. k-components
    2. PLA
    3. greedy_modularity(defalut)
    4. louvain
    5. LFM
    6. CPM
    7. COPRA
    8. SLPA
2. embedding: PLMs for encoding texts, default is *bert-base-uncased*. We use *sentence-transformers* library to get docmemnts representation, most PLMs in [huggingface](https://huggingface.co/models) could be used in g2t:
    1. *princeton-nlp/unsup-simcse-bert-base-uncased* is recommended for **English** data;
    2. *cyclone/simcse-chinese-roberta-wwm-ext* is recommened for **Chinese** data;
3. num_topics: number of topic would be showed, default is 10;
4. dim_size: dimensions would be reduced, default is 5;
## Acknowledgement
The code is implemented using [CETopic](https://github.com/hyintell/topicx)
