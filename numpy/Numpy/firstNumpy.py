import numpy as np

array = np.array([1, 2, 3, 4, 5])
print(type(array))

array += 1
print(array)

print(array.shape)

array = np.array([[1,2,3],[4,5,6]])
print(array)

fj_list = [1,2,3,4,5]
fj_array = np.array(fj_list)
print(fj_array)
#print(fj_array.dtype)  元素的类型(要求类型相同,不然会向下强制转换)

print(fj_array.itemsize) #64位,8个字节

#矩阵维度
print(fj_array.ndim)

#矩阵填充
fj_array.fill(0)

print(fj_array[1:3])

#矩阵格式(多维的形式)
fj_array = np.array([[1,2,3],
                     [4,5,6],
                     [7,8,9]])
print(fj_array.shape)

#取 5
print(fj_array[1,1])

#取第二行
print(fj_array[1])

#取第二列
print(fj_array[:,1])

print(fj_array[0,0:2])

fj_array2 = fj_array  #这时候两个变量指向同一块空间

fj_array2[1,1] = 100

print(fj_array)

fj_array2 = fj_array.copy()  #这样就不会有影响
fj_array2[1,1] = 1000
print(fj_array)

#构造等差数组  10:步长
fj_array = np.arange(0,100,10)
print(fj_array)

mask = np.array([0,0,0,2,3,1,0,1,2,0],dtype=bool)
print(mask)

#bool数组做索引
print(fj_array[mask])

random_array = np.random.rand(10)
print(random_array)

#随机数组构造bool数组
mask1 = random_array > 0.5
print(mask1)

fj_array = np.array([10,20,30,40,50])
print(np.where(fj_array > 30))  #条件索引
print(fj_array[np.where(fj_array > 30)]) #条件值