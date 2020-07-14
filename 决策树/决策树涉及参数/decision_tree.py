import matplotlib.pyplot as plt
import pandas as pd

#datasets里面包含了一些内置的数据集fetch_california_housing(房价数据集)
#对数据集解释说明的网站 http://lib.stat.cmu.edu/
from sklearn.datasets.california_housing import fetch_california_housing
#对数据进行切分成训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn import tree

import pydotplus
from IPython.display import Image

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

housing = fetch_california_housing()
#print(housing.DESCR)

#实例化数模型,指定树的最大深度为2
#参数说明:
#criterion gini  or entropy  基尼系数或者熵值
#splitter best   or random 前者是在所有特征中找到最好的切分点  后者是在部分特征中找最好切分点(使用数据量大的时候)
#max_features None(所有), log2,sqrt,N 特征小于50的时候一般使用所有的
#max_depth 指定树的最大深度(如果是2,就选择两个最好的特征)
#min_samples_split  如果某节点的样本数少于min_samples_split,则不会继续再尝试选择最优特征来进行划分如果
    #样本量不大,不需要管这个值,如果样本量数量级非常大,则推荐增大这个值
#min_samples_leaf 这个值限制了叶子节点最少的样本数,如果某叶子节点数目小于样本数,则会和兄弟节点一起被剪枝,如果
    #样本量不大,不需要管这个值
#min_weight_fraction_leaf 这个值限制了叶子节点所有样本权重和最小值
#max_leaf_nodes 通过限制最大叶子节点数,可以防止过拟合
#class_weight 指定样本各类别的权重
#min_impurity_split 这个值限制了决策树的增长
#n_estimators 要建立的树的个数
dtr = tree.DecisionTreeRegressor(max_depth=2)
#指定两列(两个特征),housing.target:结果集
dtr.fit(housing.data[:, [6,7]], housing.target)

dot_data = \
    tree.export_graphviz(
        dtr,
        out_file=None,
        feature_names=housing.feature_names[6:8],
        filled=True,
        impurity=False,
        rounded=True
    )

graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor('#FFF2DD')

Image(graph.create_png())
graph.write_png('dtr white background.png')

#random_state=42 数据重现
data_train, data_test, target_train, target_test = \
    train_test_split(housing.data, housing.target, test_size=0.1, random_state=42)

dtr = tree.DecisionTreeRegressor(random_state=42)
dtr.fit(data_train, target_train)
#
dtr.score(data_test, target_test)