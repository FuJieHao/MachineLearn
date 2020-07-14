
import pickle
import numpy as np
import os


def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        print(filename)
        datadict = pickle.load(f,encoding='iso-8859-1')
        x = datadict['data']
        y = datadict['labels']
        #对于transpose的高维数组,需要用到一个由轴编号组成的元组,才能进行转置,里面的元组的意思是对四维索引位置进行重新排序
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        y = np.array(y)

        return x, y

def load_CIFAR10(ROOT):

    xs = []
    ys = []

    for b in range(1,2):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)

    #concatenate矩阵连接
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y

    #测试集提取
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training = 5000, num_validation = 500, num_test = 500):

    cifar10_dir = 'cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    print(X_train.shape)

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    #数据的预处理

    #压缩训练集,压缩行,返回列向量,去中心化,以0为中心
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return {
        'X_train' : X_train, 'y_train' : y_train,
        'X_val' : X_val, 'y_val' : y_val,
        'X_test' : X_test, 'y_test' : y_test,
    }
