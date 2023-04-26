import random
import numpy as np
import networkx as nx
import collections
from collections import Counter

#LFM
class Community:
    """ 定义一个社区类 方便计算操作"""

    def __init__(self, G, alpha=1.0):
        self._G = G
        # α为超参数 控制观察到社区的尺度 大的α值产生非常小的社区，小的α值反而提供大的社区
        self._alpha = alpha
        self._nodes = set()
        # k_in,k_out分别代表社区总的的内部度与外部度
        self._k_in = 0
        self._k_out = 0

    def add_node(self, node):
        """ 添加节点到社区 """
        # 获得节点的邻居集  因为两点维护一边  从邻居点可以得到它的内部、外部度
        neighbors = set(self._G.neighbors(node))
        # 这里使用了集合操作简化运算 节点的k_in就等于节点在社区内的节点数(也就是节点的邻居与已经在社区中的节点集的集合)
        node_k_in = len(neighbors & self._nodes)
        # k_out自然就等于邻居数(总边数) - k_in(内部边数)
        node_k_out = len(neighbors) - node_k_in
        # 更新社区的节点、k_in、k_out
        self._nodes.add(node)
        # 对于内部度 节点本身的内部度以及在社区内的邻居节点以前的外部度变为了内部度  所以度*2
        self._k_in += 2 * node_k_in
        # 对于外部度 邻居节点在社区外  只需要计算一次 但要减去一个内部度(因为添加节点后 该节点到了社区内，以前提供的外部度变为了内部度 应该减去)
        self._k_out = self._k_out + node_k_out - node_k_in

    def remove_node(self, node):
        """ 社区去除节点 """
        neighbors = set(self._G.neighbors(node))
        # 计算与添加相反
        # community_nodes = self._nodes
        # node_k_in = len(neighbors & community_nodes)
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.remove(node)
        self._k_in -= 2 * node_k_in
        self._k_out = self._k_out - node_k_out + node_k_in

    def cal_add_fitness(self, node):
        """ 添加时计算适应度该变量 """
        neighbors = set(self._G.neighbors(node))
        old_k_in = self._k_in
        old_k_out = self._k_out
        vertex_k_in = len(neighbors & self._nodes)
        vertex_k_out = len(neighbors) - vertex_k_in
        new_k_in = old_k_in + 2 * vertex_k_in
        new_k_out = old_k_out + vertex_k_out - vertex_k_in
        # 分别用适应度公式计算
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        return new_fitness - old_fitness

    def cal_remove_fitness(self, node):
        """ 删除时计算适应度该变量 """
        neighbors = set(self._G.neighbors(node))
        new_k_in = self._k_in
        new_k_out = self._k_out
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        old_k_in = new_k_in - 2 * node_k_in
        old_k_out = new_k_out - node_k_out + node_k_in
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self._alpha
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self._alpha
        return new_fitness - old_fitness

    def recalculate(self):
        # 遍历社区中是否有适应度为负的节点
        for vid in self._nodes:
            fitness = self.cal_remove_fitness(vid)
            if fitness < 0.0:
                return vid
        return None

    def get_neighbors(self):
        """ 获得社区的邻居节点 方便后面遍历 """
        neighbors = set()
        # 统计社区内所有节点的邻居，其中不在社区内部的邻居节点 就是社区的邻居节点
        for node in self._nodes:
            neighbors.update(set(self._G.neighbors(node)) - self._nodes)
        return neighbors

    def get_fitness(self):
        return float(self._k_in) / ((self._k_in + self._k_out) ** self._alpha)


class LFM:

    def __init__(self, G, alpha):
        self._G = G
        # α为超参数 控制观察到社区的尺度 大的α值产生非常小的社区，小的α值反而提供大的社区
        self._alpha = alpha

    def execute(self):
        communities = []
        # 统计还没被分配到社区的节点(初始是所有节点)
        # node_not_include = self._G.node.keys()[:]
        node_not_include = list(self._G.nodes())
        while len(node_not_include) != 0:
            # 初始化一个社区
            c = Community(self._G, self._alpha)
            # 随机选择一个种子节点
            seed = random.choice(node_not_include)
            # print(seed)
            c.add_node(seed)

            # 获得社区的邻居节点并遍历
            to_be_examined = c.get_neighbors()
            while to_be_examined:
                # 添加适应度最大的节点到社区
                m = {}
                for node in to_be_examined:
                    fitness = c.cal_add_fitness(node)
                    m[node] = fitness
                to_be_add = sorted(m.items(), key=lambda x: x[1], reverse=True)[0]

                # 当所有节点适应度为负  停止迭代
                if to_be_add[1] < 0.0:
                    break
                c.add_node(to_be_add[0])

                # 遍历社区中是否有适应度为负的节点 有则删除
                to_be_remove = c.recalculate()
                while to_be_remove is not None:
                    c.remove_node(to_be_remove)
                    to_be_remove = c.recalculate()

                to_be_examined = c.get_neighbors()

            # 还没被分配到社区的节点集中删除已经被添加到社区中的节点
            for node in c._nodes:
                if node in node_not_include:
                    node_not_include.remove(node)
            communities.append(c._nodes)
        return communities



#CPM (clique percolation method)
# from networkx.algorithms.community import k_clique_communities

#COPRA
class COPRA:
    def __init__(self, G, T, v):
        """
        :param G:图本身
        :param T: 迭代次数T
        :param r:满足社区次数要求的阈值r
        """
        self._G = G
        self._n = len(G.nodes(False))  # 节点数目
        self._T = T
        self._v = v

    def execute(self):
        # 建立成员标签记录
        # 节点将被分配隶属度大于阈值的社区标签
        lablelist = {i: {i: 1} for i in self._G.nodes()}
        for t in range(self._T):
            visitlist = list(self._G.nodes())
            # 随机排列遍历顺序
            np.random.shuffle(visitlist)
            # 开始遍历节点
            for visit in visitlist:
                temp_count = 0
                temp_label = {}
                total = len(self._G[visit])
                # 根据邻居利用公式计算标签
                for i in self._G.neighbors(visit):
                    res = {key: value / total for key, value in lablelist[i].items()}
                    temp_label = dict(Counter(res) + Counter(temp_label))
                temp_count = len(temp_label)
                temp_label2 = temp_label.copy()
                for key, value in list(temp_label.items()):
                    if value < 1 / self._v:
                        del temp_label[key]
                        temp_count -= 1
                # 如果一个节点中所有的标签都低于阈值就随机选择一个
                if temp_count == 0:
                    # temp_label = {}
                    # v = self._v
                    # if self._v > len(temp_label2):
                    #     v = len(temp_label2)
                    # b = random.sample(temp_label2.keys(), v)
                    # tsum = 0.0
                    # for i in b:
                    #     tsum += temp_label2[i]
                    # temp_label = {i: temp_label2[i]/tsum for i in b}
                    b = random.sample(temp_label2.keys(), 1)
                    temp_label = {b[0]: 1}
                # 否则标签个数一定小于等于v个 进行归一化即可
                else:
                    tsum = sum(temp_label.values())
                    temp_label = {key: value / tsum for key, value in temp_label.items()}
                lablelist[visit] = temp_label

        communities = collections.defaultdict(lambda: list())
        # 扫描lablelist中的记录标签，相同标签的节点加入同一个社区中
        for primary, change in lablelist.items():
            for label in change.keys():
                communities[label].append(primary)
        # 返回值是个数据字典，value以集合的形式存在
        return communities.values()
#SLPA
class SLPA:
    def __init__(self, G, T, r):
        """
        :param G:图本省
        :param T: 迭代次数T
        :param r:满足社区次数要求的阈值r,越小发现的重叠社区越多
        """
        self._G = G
        self._n = len(G.nodes(False))  # 节点数目
        self._T = T
        self._r = r

    def execute(self):
        # 将图中数据录入到数据字典中以便使用
        weight = {j: {} for j in self._G.nodes()}
        for q in weight.keys():
            for m in self._G[q].keys():
                # weight[q][m] = self._G[q][m]['weight']
                weight[q][m] = 1
        # 建立成员标签记录  初始本身标签为1
        memory = {i: {i: 1} for i in self._G.nodes()}
        # 开始遍历T次所有节点
        for t in range(self._T):
            listenerslist = list(self._G.nodes())
            # 随机排列遍历顺序
            np.random.shuffle(listenerslist)
            # 开始遍历节点
            for listener in listenerslist:
                # 每个节点的key就是与他相连的节点标签名
                # speakerlist = self._G[listener].keys()
                labels = collections.defaultdict(int)
                # 遍历所有与其相关联的节点
                for speaker in self._G.neighbors(listener):
                    total = float(sum(memory[speaker].values()))
                    # 查看speaker中memory中出现概率最大的标签并记录，key是标签名，value是Listener与speaker之间的权
                    # multinomial从多项式分布中提取样本。
                    # 多项式分布是二项式分布的多元推广。做一个有P个可能结果的实验。这种实验的一个例子是掷骰子，结果可以是1到6。
                    # 从分布图中提取的每个样本代表n个这样的实验。其值x_i = [x_0，x_1，…，x_p] 表示结果为i的次数。
                    # 函数语法
                    # numpy.random.multinomial(n, pvals, size=None)
                    #
                    # 参数
                    # n :  int：实验次数
                    # pvals：浮点数序列，长度p。P个不同结果的概率。这些值应该和为1（但是，只要求和（pvals[：-1]）<=1，最后一个元素总是被假定为考虑剩余的概率）。
                    # size :  int 或 int的元组，可选。 输出形状。如果给定形状为（m，n，k），则绘制 m*n*k 样本。默认值为无，在这种情况下返回单个值。
                    labels[list(memory[speaker].keys())[
                        np.random.multinomial(1, [freq / total for freq in memory[speaker].values()]).argmax()]] += \
                    weight[listener][speaker]
                # 查看labels中值最大的标签，让其成为当前listener的一个记录
                maxlabel = max(labels, key=labels.get)
                if maxlabel in memory[listener]:
                    memory[listener][maxlabel] += 1
                else:
                    memory[listener][maxlabel] = 1.5
        # 提取出每个节点memory中记录标签出现最多的一个
        # for primary in memory:
        #     p = list(memory[primary].keys())[
        #         np.random.multinomial(1, [freq / total for freq in memory[primary].values()]).argmax()]
        #     memory[primary] = {p: memory[primary][p]}
        n_overlap=0
        for m in memory.values():
            sum_label = sum(m.values())
            threshold_num = sum_label * self._r
            for k, v in list(m.items()):
                if v < threshold_num:
                    del m[k]
            if len(m) >1:
                n_overlap+=1

        communities = collections.defaultdict(lambda: list())
        # 扫描memory中的记录标签，相同标签的节点加入同一个社区中
        for primary, change in memory.items():
            for label in change.keys():
                communities[label].append(primary)
        # 返回值是个数据字典，value以集合的形式存在
        print("no of overlap nodes: "+str(n_overlap))
        return communities.values()
