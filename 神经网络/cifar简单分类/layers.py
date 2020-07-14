import numpy as np

def affine_forward(x, w, b):

    out = None
    N = x.shape[0]
    #这里的-1是模糊控制的意思,就是说,把x变成 N行 自动匹配的列
    x_row = x.reshape(N, - 1)
    out = np.dot(x_row, w) + b
    cache = (x, w, b)

    return out, cache

#dout 上一层传下来的梯度
def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)
    x_row = x.reshape(x.shape[0], -1)
    dw = np.dot(x_row.T, dout)
    db = np.sum(dout, axis=0, keepdims=True)

    return dx, dw, db


def relu_forward(x):

    out = None
    out = ReLu(x)
    cache = x
    return out, cache

def relu_backward(dout, cache):

    dx, x = None, cache
    dx = dout
    dx[x <= 0] = 0

    return dx

def softmax_loss(x, y):
    '''

    :param x: 得分值
    :param y: 标签值
    :return: 损失值,
    '''

    #归一化操作,把传进来的x得分值转换成概率值
    #np.max中, axis = 1,表是对行矩阵方向求最大值, keepdims = True表示保持其二维特性
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))  #归一化
    probs /= np.sum(probs, axis=1, keepdims=True)       #转换成概率值

    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N #平均损失值

    #反向传播的时候求导
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    #返回损失值,梯度
    return loss, dx

def ReLu(x):
    return np.maximum(0, x)