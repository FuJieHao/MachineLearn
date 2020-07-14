import numpy as np

fj_array = np.array([[1.5,1.3,7.5],
                    [5.6,7.8,1.2]])

#排序
print(np.sort(fj_array))

#索引值排序(原来索引值现在的位置)
print(np.argsort(fj_array))

#按照平均梯度构建一个数组
fj_array = np.linspace(0,10,10)
print(fj_array)

values = np.array([2.5,6.5,9.5])
#数组插值(待插值的数组必须是排序好的)
print(np.searchsorted(fj_array,values))


fj_array = np.array([[1,0,6],
                    [1,7,0],
                    [2,3,1],
                    [2,4,0]])
print(fj_array)

#对最后一列升序排列,对第一列降序排列
index = np.lexsort([-1 * fj_array[:,0], fj_array[:,2]])
print(index)

fj_array = fj_array[index]
print(fj_array)