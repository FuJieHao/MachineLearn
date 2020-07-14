import tensorflow as tf
import numpy as np

#创建变量
w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0],[1.0]])

y = tf.matmul(x, w)

#初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(y.eval())

#3,4矩阵
#tf.zeros([3,4], float)
#tf.zeros_like()
#tf.ones
#tf.ones_like()

#把常量转换成tf支持的格式
#tensor = tf.constant([1,2,3])

#tensor = tf.constant(-1.0, shape=[2,3])
# ==> [[-1 -1 -1] [-1 -1 -1]]

#tf.linspace(10.0, 12.0, 3, name='linspace')
# ==> [10.0 11.0 12.0]

#start = 3
#limit = 18
#delta = 3
#tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

#随机矩阵 ,高斯分布,      ,mean-1 均值  stddev方差
#norm = tf.random_normal([2, 3], mean=-1, stddev=4)

#矩阵洗牌
#c = tf.constant([[1, 2], [3, 4], [5, 6]])
#shuff = tf.random_shuffle(c)
#(推荐使用 with tf.Session() as sess:这个结构) 用这个结构的话要初始化
'''
with tf.Session() as sess:
    sess.run(init_op)
    print(shuff.eval())
'''
#sess = tf.Session()
#print(sess.run(shuff))

'''
state = tf.Variable(0)
new_value = tf.add(state, tf.constant(1))
#把new_value 赋值 给 state
update = tf.assign(state, new_value)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in  range(3):
        sess.run(update)
        print(sess.run(state))
'''

'''
#把numpy 转换成 tensorflow
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))
'''

#占位placeholder ,在 session 中占位
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    #在run的时候进行赋值
    print(sess.run([output], feed_dict={input1 : [7.], input2:[2.]}))
