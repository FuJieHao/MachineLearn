import numpy as np

x = np.array([5,5])
y = np.array([2,2])

#对应位置相乘
print(np.multiply(x,y)) #[10 10]

#矩阵乘法,同维  #20
print(np.dot(x,y))

x.shape = 2,1
print(x)
y.shape = 1,2
print(y)

print(np.dot(x,y))

print(np.dot(y,x))

#当维度不同的时候,自动补维
x = np.array([1,1,1])
y = np.array([[1,2,3],[4,5,6]])
print(x * y)

x = np.array([1,1,1])
y = np.array([1,1,1])
#维度一样,逐一进行比较
print(x == y)

#真假假,真真真,假假假  与
print(np.logical_and(x,y))
#或
print(np.logical_or(x,y))
#非
np.logical_not(x,y)