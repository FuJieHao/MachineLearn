import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import time

from sklearn import preprocessing as pp

path = 'data' + os.sep + 'LogiReg_data.txt'
pdData = pd.read_csv(path, header = None, names = ['Exam 1', 'Exam 2','Admitted'])
print(pdData.head())
print(pdData.shape)
#
# #确定的
# positive = pdData[pdData['Admitted'] == 1]
# #否定的
# negative = pdData[pdData['Admitted'] == 0]
#
# fig,ax = plt.subplots(figsize = (10,5))
# #画散点图
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s = 30, c = 'b', marker = 'o', label = 'Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s = 30, c = 'r', marker = 'x', label = 'Not Admitted')
#
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
#
# plt.show()

#目标:建立分类器
#通过设定阈值,根据阈值判断录取结果

#sigmoid函数
#自变量取值为任意实数,值域[0,1],完成值到概率的映射
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#返回预测函数值
# X:数据(1,特征值1,特征值2,...)
#theta:参数值
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))

#添加一列,0:列的位置,列名:ones,值:1
pdData.insert(0,'Ones',1)
#矩阵化
orig_data = pdData.as_matrix()
#矩阵的列数
cols = orig_data.shape[1]
#:,0:cols - 1   :前是对于行的操作,后面是对列的操作
X = orig_data[:,0:cols - 1]
#最后一列
y = orig_data[:,cols-1:cols]
#一行三列填充0
theta = np.zeros([1,3])

#X:数据, y:标签 theta:参数
#y:作用是统一化表达式,因为是分类的缘故,用1-y代表0,y代表1
#这里直接用对数似然进行运算,而没有使用最小二乘法
def cost(X, y, theta):
    left  = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    #求个平均损失
    return np.sum(left - right) / (len(X))

print(cost(X, y, theta))

#梯度计算
#通过对似然函数的化简发现(似然函数越大越好),那么(目标函数)最小二乘越小越好
#让其偏导等于0即可,也就是梯度下降
def gradient(X, y, theta):
    #求几个方向的梯度
    grad  = np.zeros(theta.shape)
    #ravel 的 作用是 降维
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:,j])
        grad[0,j] = np.sum(term) / len(X)

    return grad

#比较3种不同梯度下降的方法
STOP_ITER = 0   #按照迭代次数停止
STOP_COST = 1   #按照损失值停止
STOP_GRAD = 2   #按照梯度停止

def stopCriterion(type, value, threshold):
    #设定三种不同的停止策略
    #threshold:次数
    if type   == STOP_ITER:   return value > threshold
    #threshold:阈值
    elif type == STOP_COST:   return abs(value[-1] - value[-2]) < threshold
    #threshold:阈值
    elif type == STOP_GRAD:   return np.linalg.norm(value) < threshold

#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0 : cols - 1]
    y = data[:, cols - 1 :]
    return X,y


#进行参数更新
#thresh 策略对应的阈值 alpha 学习率
def descent(data, theta, batchSize, stopType, thresh, alpha):
    #梯度下降求解
    init_time = time.time()
    i = 0 #迭代次数
    k = 0 #batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape) #计算的梯度
    costs = [cost(X, y, theta)] #损失值

    while True:
        grad = gradient(X[k : k + batchSize], y[k : k + batchSize], theta)
        k += batchSize #取batch数量个数据
        if k >= data.shape[0]:
            k = 0
            X, y = shuffleData(data) #重新洗牌
        theta = theta - alpha * grad #参数更新
        costs.append(cost(X, y, theta)) #计算新的损失值,拼接到list
        i += 1

        if stopType   == STOP_ITER:   value = i
        elif stopType == STOP_COST:   value = costs
        elif stopType == STOP_GRAD:   value = grad
        if stopCriterion(stopType, value, thresh): break

    return theta, i - 1, costs, grad, time.time() - init_time

def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize == data.shape[0]: strDescType = "Gradient"
    elif batchSize == 1:  strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()
    return theta

n = 100
#按照迭代次数,耗时1s多,损失值0.63
#runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)

#按照损失值,小的学习率,耗时25.97s,损失值0.38
#runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)

#按照梯度,阈值0.05,小的学习率,耗时8.42s,损失值0.49
#runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)

#采取单个样本,迭代5000次的策略,耗时0.33s,损失值1.29(波动太大)
#runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)

#单个样本,扩大学习次数,降低学习率,耗时0.9s,损失值0.63
#runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000002)

#小批量样本,稍微大一点的学习率,耗时1.23s,损失值0.78
#runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)

#对数据进行标准化处理,将数据按照列减去其均值,然后除以方差
#最后可以得到的结果是,对每个属性(每列)来说所有的数据都聚集在0的附近,方差为1
scaled_data = orig_data.copy()
scaled_data[:, 1:3] = pp.scale(orig_data[:, 1:3])
#全部样本,小的迭代次数,教高的学习率,耗时1.27s,损失值0.38
#runExpe(scaled_data, theta, n, STOP_ITER, thresh=5000, alpha=0.001)

#按照梯度,阈值0.02
#runExpe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)

#单个样本,按照梯度,较小的阈值,耗时5.41s,损失值0.22
theta = runExpe(scaled_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.001)

#小批量样本,耗时6.62s,损失值0.22
#runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)

def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]

scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0))
           else 0 for (a,b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print("accuracy = {0}%".format(accuracy))