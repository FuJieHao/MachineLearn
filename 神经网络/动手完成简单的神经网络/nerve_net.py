import numpy as np

def sigmoid(x, deriv = False):
    if (deriv == True):
        return x * (1 - x)  #这里的x相当于整个sigmoid函数

    return 1 / (1 + np.exp(-x))

x = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1],
             [0,0,1]]
)

print(x.shape)

#标签,类别
y = np.array([[0],
              [1],
              [1],
              [0],
              [0]
              ])

print(y.shape)

np.random.seed(1)

#构造一个随机矩阵,3行4列 (3行和输入相连(特征))
#4代表后面连接4个神经元

#指定为 - 1 到 1 的区间上
w0 = 2 * np.random.random((3,4)) - 1

w1 = 2 * np.random.random((4,1)) - 1

print(w0)

#进行迭代
for j in range(60000):
    l0 = x
    #进行激活函数,前向传播
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))

    #进行误差项分析
    l2_error = y - l2
    if(j % 10000) == 0:
        print('Error' + str(np.mean(np.abs(l2_error))))
    #l2_error作为权重项
    l2_delta = l2_error * sigmoid(l2, deriv=True)
    l1_error = l2_delta.dot(w1.T)

    l1_delta = l1_error * sigmoid(l1, deriv=True)

    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)
