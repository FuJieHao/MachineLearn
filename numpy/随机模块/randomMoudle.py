import numpy as np

#从0到1的3行2列随机矩阵
print(np.random.rand(3,2))

#0到10的随机值,5行4列  区间:[)
print(np.random.randint(10,size=(5,4)))

#一个随机浮点数
print(np.random.rand())
#np.random.random_sample() 同上

#0到10 3个整型数
print(np.random.randint(0,10,3))

#精度,小数点后三位
np.set_printoptions(precision= 3)

#高斯分布
mu, sigma = 0,0.1
print(np.random.normal(mu,sigma,10))

#洗牌
fj_array = np.arange(10)
np.random.shuffle(fj_array)
print(fj_array)

#随机种子 (保证调节参数,唯一变量)
np.random.seed(0)

mu, sigma = 0,0.1
print(np.random.normal(mu,sigma,10))