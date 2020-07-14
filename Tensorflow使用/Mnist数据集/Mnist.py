import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

mnist = input_data.read_data_sets('data/', one_hot=True)

'''
784代表像素点的个数
print (" tpye of 'mnist' is %s" % (type(mnist)))
print (" number of trian data is %d" % (mnist.train.num_examples))
print (" number of test data is %d" % (mnist.test.num_examples))

print ("What does the data of MNIST look like?")
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print
print (" type of 'trainimg' is %s"    % (type(trainimg)))
print (" type of 'trainlabel' is %s"  % (type(trainlabel)))
print (" type of 'testimg' is %s"     % (type(testimg)))
print (" type of 'testlabel' is %s"   % (type(testlabel)))
print (" shape of 'trainimg' is %s"   % (trainimg.shape,))
print (" shape of 'trainlabel' is %s" % (trainlabel.shape,))
print (" shape of 'testimg' is %s"    % (testimg.shape,))
print (" shape of 'testlabel' is %s"  % (testlabel.shape,))
'''

training   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels

'''
print ("How does the training data look like?")
nsample = 5
randidx = np.random.randint(trainimg.shape[0], size=nsample)

for i in randidx:
    curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix
    curr_label = np.argmax(trainlabel[i, :] ) # Label
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title("" + str(i) + "th Training Data "
              + "Label is " + str(curr_label))
    print ("" + str(i) + "th Training Data "
           + "Label is " + str(curr_label))
    plt.show()
'''

#这里的 None 是无穷的意思
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#模型的建立 softmax 预测值的结果
actv = tf.nn.softmax(tf.matmul(x, W) + b)

#计算损失值
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))
#指定学习率
learning_rate = 0.01
#梯度下降优化器
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#求矩阵中每一列的最大值的索引  0:列 1:行
#tf.argmax(arr, 0).eval()

#将预测值的索引和真实值的索引进行比较 true flase
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))

#求均值 cast: 把pred转换成float类型
accr = tf.reduce_mean(tf.cast(pred, 'float'))

#初始化
init = tf.global_variables_initializer()

#初始化变量
training_epochs = 50
batch_size = 100
display_step = 5

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0
    num_batch = int(mnist.train.num_examples / batch_size)
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm, feed_dict={x : batch_xs, y : batch_ys})
        feeds = {x : batch_xs, y : batch_ys}
        #平均损失值
        avg_cost += sess.run(cost, feed_dict=feeds) / num_batch

    #每迭代五次打印一下
    if epoch % display_step == 0:
        feeds_train = {x:batch_xs, y:batch_ys}
        feeds_test = {x : mnist.test.images, y : mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
              % (epoch, training_epochs, avg_cost, train_acc, test_acc))

print("Done")
