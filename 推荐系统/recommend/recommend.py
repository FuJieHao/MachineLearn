from  __future__ import (absolute_import, division, print_function, unicode_literals)
#KNNBasic  最基础的协同过滤算法, SVD 矩阵分解的方式
from surprise import KNNBasic, SVD
#
from surprise import Dataset

from surprise import KNNBaseline

#from surprise import evaluate, print_perf
from surprise import model_selection

#类似于scikit-learn

#gridsearch 是为了解决调参的问题,找出最优化的参数
from surprise import GridSearch

import os
import io
import pandas as pd

from collections import deque
from six import next

import tensorflow as tf
import numpy as np

import time
import readers



'''
#数据下载
data = Dataset.load_builtin('ml-1m')


#交叉验证
data.split(n_folds=3)

#实例化算法对象
algo = KNNBasic()

#评估模块,  算法,   数据,   评估标准(均方误差,绝对误差)
perf = evaluate(algo, data, measures=['RMSE','MAE'])

print_perf(perf)
'''

'''
#指定参数: 迭代参数     学习率     正则化惩罚的
param_grid = {'n_epochs':[5, 10], 'lr_all':[0.002, 0.005],
              'reg_all':[0.4, 0.6]}

#SVD算法, 参数, 衡量标准
#grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])
grid_search = model_selection.GridSearchCV(SVD, param_grid, measures=['RMSE', 'FCP'])
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

grid_search.fit(data)

print(grid_search.best_params)

results_df = pd.DataFrame.from_dict(grid_search.cv_results)
print(results_df)
'''


'''
def read_item_names():

    file_name = ('./ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid

data = Dataset.load_builtin('ml-100k')
#转换成标准矩阵(稀疏)
trainset = data.build_full_trainset()

#计算相似度,皮尔逊相似度算法
sim_options = {'name':'pearson_baseline', 'user_based':False}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

rid_to_name, name_to_rid = read_item_names()

#对应的ID
toy_story_raw_id = name_to_rid['Now and Then (1995)']
print(toy_story_raw_id)

#在矩阵中的位置
toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
print(toy_story_inner_id)

#求近邻
toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k = 10)
print(toy_story_neighbors)

#找出近邻对应的电影的名字
toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
                       for inner_id in toy_story_neighbors)
toy_story_neighbors = (rid_to_name[rid]
                       for rid in toy_story_neighbors)

print('The 10 nearest neighbors of Toy Story are:')
for movie in toy_story_neighbors:
    print(movie)
'''


np.random.seed(42)

#用户数
u_num = 6040
#电影数
i_num = 3952

#迭代的更新的大小
batch_size = 1000

#隐含因子的维度
dims = 5

#迭代的次数
max_epochs = 50

#选择运算的是cpu 还是 gpu
place_device = '/cpu:0'

def get_data():
    #./ml-1m/ratings.dat

    df = readers.read_file('./ml-1m/ml-1m/ratings.dat', sep='::')
    rows = len(df)

    #洗牌
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)

    #分割数据集
    split_index = int(rows * 0.9)

    df_train = df[0 : split_index]
    df_test = df[split_index :].reset_index(drop=True)

    return df_train, df_test

def clip(x):
    return np.clip(x, 1.0, 5.0)

def model(user_batch, item_batch, user_num, item_num, dim = 5, device = '/cpu:0'):
    with tf.device('/cpu:0'):
        with tf.variable_scope('lsi', reuse=tf.AUTO_REUSE):

            #添加一个全局的大环境
            bias_global = tf.get_variable("bias_global", shape=[])

            #指定user_bias 和 item_bias
            #embd_bias_user 共享操作
            w_bias_user = tf.get_variable('embd_bias_user', shape=[user_num])
            w_bias_item = tf.get_variable('embd_bias_item', shape=[item_num])

            #选择迭代样本数的部分:lookup进行查找工作
            bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name='bias_user')
            bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name='bias_item')

            #指定优化的矩阵
            w_user = tf.get_variable('embd_user', shape=[user_num, dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            w_item = tf.get_variable('embd_item', shape=[item_num, dim],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            embd_user = tf.nn.embedding_lookup(w_user, user_batch, name='embedding_user')
            embd_item = tf.nn.embedding_lookup(w_item, item_batch, name='embedding_item')

        with tf.device(device):
            infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
            infer = tf.add(infer, bias_global)
            infer = tf.add(infer, bias_user)
            infer = tf.add(infer, bias_item, name='svd_inference')
            #添加正则化惩罚项,防止出现过拟合的现象
            regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item),
                                 name='svd_regularizer')

        return infer, regularizer

def loss(infer, regularizer, rate_batch, learning_rate = 0.001, reg = 0.1, device = '/cpu:0'):
    with tf.device(device):
        #预测值和真实值之间的差异
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))

        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name='l2')
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        #进行梯度下降
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    return cost, train_op

df_train, df_test = get_data()

samples_per_batch = len(df_train) // batch_size
#数据规模
print('Number of train samples %d, test samples %d,'
      'samples per batch %d' % (len(df_train), len(df_test),
                                samples_per_batch))


#进行迭代训练

#洗牌
iter_train = readers.ShuffleIterator([df_train['user'],
                                      df_train['item'],
                                      df_train['rate']],
                                     batch_size=batch_size)

#做 一次 迭代测试
iter_test = readers.OneEpochIterator([df_test['user'],
                                      df_test['item'],
                                      df_test['rate']],
                                     batch_size=-1)


user_batch = tf.placeholder(tf.int32, shape=[None], name='id_user')
item_batch = tf.placeholder(tf.int32, shape=[None], name='id_item')
rate_batch = tf.placeholder(tf.float32, shape=[None])

infer, regularizer = model(user_batch, item_batch, user_num=u_num,
                           item_num=i_num, dim=dims, device=place_device)
_, train_op = loss(infer, regularizer, rate_batch, learning_rate=0.0010, reg=0.05, device=place_device)

saver = tf.train.Saver()
#初始化全局变量
init_op = tf.global_variables_initializer()


with tf.Session(config=tf.ConfigProto(device_count = {'CPU':4})) as sess:
    sess.run(init_op)
    print('%s\t%s\t%s\t%s' % ('Epoch', 'Train Error', 'Val Error', 'Elapsed Time'))
    errors = deque(maxlen=samples_per_batch)
    start = time.time()
    for i in range(max_epochs * samples_per_batch):
        users, items, rates = next(iter_train)
        _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                               item_batch: items,
                                                               rate_batch: rates})

        pred_batch = clip(pred_batch)
        errors.append(np.power(pred_batch - rates, 2))
        if i % samples_per_batch == 0:
            train_err = np.sqrt(np.mean(errors))
            test_err2 = np.array([])
            for users, items, rates in iter_test:
                pred_batch = sess.run(infer, feed_dict= {user_batch: users,
                                                         item_batch: items})
                pred_batch = clip(pred_batch)
                test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))

            end = time.time()

            print('%02d\t%.3f\t\t%.3f\t\t%.3f secs' % (i // samples_per_batch, train_err,
                                                       np.sqrt(np.mean(test_err2)), end - start))

            start = end

        saver.save(sess, './save/')





