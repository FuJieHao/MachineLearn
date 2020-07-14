from layer_utils import *
import numpy as np

class TwoLayerNet(object):
    def __init__(self, input_dim = 3 * 32 * 32, hidden_dim = 100,
                 num_classes = 10, weight_scale = 1e-3, reg = 0.0):
        '''
        :param input_dim: 输入的维度
        :param hidden_dim: 中间隐层的神经元数
        :param num_classes: 最终分类的类别
        :param weight_scale: 权重初始化的小值
        :param reg: 正则化惩罚权重项(力度)
        :return:
        '''

        self.params = {}
        self.reg = reg
        #前后连接
        self.params['w1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros((1, hidden_dim))
        self.params['w2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros((1, num_classes))

    #定义损失函数, y 是标签值
    def loss(self, X, y = None):
        scores = None
        N = X.shape[0]

        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']

        #中间有一个relu层
        h1, cache1 = affine_relu_forward(X, w1, b1)
        #输出的时候不需要relu层,直接前向传播
        out, cache2 = affine_forward(h1, w2, b2)
        scores = out

        #如果y是空的话,这就是我们正在使用测试集,那么只需要返回得分值
        if y is None:
            return scores

        #grads存梯度的值
        loss, grads = 0, {}
        #softmax分类器
        data_loss, dscores = softmax_loss(scores, y)
        #正则化惩罚项
        reg_loss = 0.5 * self.reg * np.sum(w1 * w1) + 0.5 * self.reg * np.sum(w2 * w2)
        #现在损失值 = 损失值 + 正则化损失值
        loss = data_loss + reg_loss

        dh1, dw2, db2 = affine_backward(dscores, cache2)
        dx, dw1, db1 = affine_relu_backward(dh1, cache1)

        #进行权重更新
        dw2 += self.reg * w2
        dw1 += self.reg * w1

        grads['w1'] = dw1
        grads['b1'] = db1
        grads['w2'] = dw2
        grads['b2'] = db2

        return loss, grads






