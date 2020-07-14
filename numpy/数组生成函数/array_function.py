import numpy as np

np.array([1,2,3])

np.arange(10)

#从哪到哪,步长
np.arange(2,20,2)

#指定为浮点32
np.arange(2,20,2,dtype=np.float32)

#从0到10构造50个数
print(np.linspace(0,10,50))

#默认是以10为底,5个数
print(np.logspace(0,1,5))

x = np.linspace(-10,10,5)
y = np.linspace(-10,10,5)

#构造网格,变成二维结构
x,y = np.meshgrid(x,y)

#构造行向量
print(np.r_[0:10:1])
#构造列向量
print(np.c_[0:10:1])

