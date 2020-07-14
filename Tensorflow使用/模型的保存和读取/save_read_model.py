import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
training = mnist.train.images
trainlabel = mnist.train.labels
testing = mnist.test.images
testlabel = mnist.test.labels
print("MNIST ready")


n_input = 784
n_output = 10

weights = {
    #random_normal : 高斯初始化
    #卷积层参数 3:filter height  3: width   1: deep   64: output特征图
    'wc1' : tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    'wc2' : tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
    #全连接层参数 7*7*128: 两层2*2池化之后(28 变成 7)
    'wd1' : tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev=0.1)),
    'wd2' : tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
}
#偏置项初始化 wx + b
biases = {
    'bc1' : tf.Variable(tf.random_normal([64], stddev=0.1)),
    'bc2' : tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd1' : tf.Variable(tf.random_normal([1024], stddev=0.1)),
    'bd2' : tf.Variable(tf.random_normal([n_output], stddev=0.1))
}

def conv_basic(_input, _w, _b, _keepratio):
    #输入
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    #第一层卷积 , SAME会在移动中自动填充0
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    #添加激活函数
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    #池化层 ,ksize 指定窗口大小,[batch,h,w,deep]
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #随机丢弃一定量的节点,   _keepratio保留的节点比例
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)

    #第二层卷积
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)

    #对输出reshape一下
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])

    #第一个全连接层
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)

    #第二个全连接层
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    #返回值
    out = {'input_r' : _input_r, 'conv1' : _conv1, 'pool1' : _pool1,
           'pool1_dr1' : _pool_dr1, 'conv2' : _conv2, 'pool2' : _pool2,
           'pool_dr2' : _pool_dr2, 'dense1' : _dense1, 'fc1' : _fc1,
           'fc_dr1' : _fc_dr1, 'out' : _out}

    return out

print('CNN READY')

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

_pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_pred, labels=y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.global_variables_initializer()

# SAVER
#保存间隔,一个epoch一保存
save_step = 1
#max_to_keep = 3 值保留3组模型
saver = tf.train.Saver(max_to_keep=3)

print('GRAPH READY')

do_train = 0
sess = tf.Session()
sess.run(init)

training_epochs = 15
batch_size = 16
display_step = 1

#训练模块
if do_train == 1:
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = 10

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run(optm, feed_dict={x : batch_xs, y : batch_ys, keepratio : 0.7})
            avg_cost += sess.run(cost, feed_dict={x : batch_xs, y : batch_ys, keepratio : 1.}) / total_batch

        if epoch % display_step == 0:
            print('Epoch : %03d / %03d cost : %.9f' % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x : batch_xs, y : batch_ys, keepratio : 1.})
            #accuracy 精确
            print('Training accuracy : %.3f' % train_acc)

        #保存训练网络
        if epoch % save_step == 0:
            saver.save(sess, 'save/nets/cnn_mnist_basic.ckpt-' + str(epoch))

print('OPTIMIZATION FINISHED')

#测试模块
if do_train == 0:
    epoch = training_epochs - 1
    saver.restore(sess, 'save/nets/cnn_mnist_basic.ckpt-' + str(epoch))

    test_acc = sess.run(accr, feed_dict={x : testing, y : testlabel, keepratio : 1.})
    print('TEST ACCURACY : %.3f' % test_acc)


#简单模型保存
'''
v1 = tf.Variable(tf.random_normal([1,2]), name='v1')
v2 = tf.Variable(tf.random_normal([2,3]), name='v2')

init_op = tf.global_variables_initializer()

#初始化saver对象
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(init_op)
    print('V1', sess.run(v1))
    print('V2', sess.run(v2))
    saver_path = saver.save(sess, 'save/model.ckpt')
    print('Model saved in file :', saver_path)


with tf.Session() as sess:
    saver.restore(sess, 'save/model.ckpt')
    print('V1:', sess.run(v1))
    print('V2:', sess.run(v2))
    print('Model restored')
'''