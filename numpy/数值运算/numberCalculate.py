import numpy as np

fj_array = np.array([[1,2,3],[4,5,6]])

#数组求和
print(np.sum(fj_array))

#指定要进行的操作是沿着什么维度,列求和
print(np.sum(fj_array,axis=0))
#行求和
print(np.sum(fj_array,axis=1))

print(np.sum(fj_array,axis=-1))

print(fj_array.sum(axis=1))

#数组乘积
print(fj_array.prod())

print(fj_array.prod(axis=0))

#全局最小值
print(fj_array.min())
print(fj_array.min(axis=1))

#最小值索引位置
print(fj_array.argmin())

print(fj_array.argmax(axis=0))

#均值
print(fj_array.mean())

#标准差(也可指定维度)
print(fj_array.std())
#方差(也可指定维度)
print(fj_array.var())

#限制元素的范围(小的到2,大的到4)
print(fj_array.clip(2,4))

fj_array = np.array([1.2,3.56,3.13])
#四舍五入
print(fj_array.round())
#指定精度的四舍五入
print(fj_array.round(decimals=1))
