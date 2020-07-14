import numpy as np

#构造长度为3的空向量(填充0)
print(np.zeros(3))

print(np.zeros((3,3)))

#构造填充2的矩阵
print(np.ones((3,3)) * 2)

np.ones((3,3),dtype=np.float32)

#构造空矩阵
a = np.empty(6)
#填充
a.fill(2)

fj_array = np.array([1,2,3,4])
#构造一个大小相同的矩阵
print(np.zeros_like(fj_array))
np.ones_like(fj_array)

#构造单位矩阵
print(np.identity(5))