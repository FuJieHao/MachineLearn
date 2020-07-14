#对数组形状进行操作
import numpy as np

fj_array = np.arange(10)
print(fj_array)

print(fj_array.shape)

#更改维度   改为2维,
fj_array.shape = 2,5
print(fj_array)

#改变维度
print(fj_array.reshape(1,10))

fj_array = np.arange(10)
fj_array = fj_array[np.newaxis,:]
print(fj_array)   #二维(1,10)

fj_array = np.arange(10)
fj_array = fj_array[:,np.newaxis]
print(fj_array.shape) #(10,1)

fj_array = fj_array[:,np.newaxis,np.newaxis]
print(fj_array.shape)

#空轴压缩
fj_array = fj_array.squeeze()
print(fj_array.shape)


fj_array.shape = 2,5
#转置
print(fj_array.transpose())

# 另一种转置 print(fj_array.T)

#数组的连接
a = np.array([[123,456,678],[3214,456,134]])
b = np.array([[1235,3124,432],[43,13,134]])

#里面放一个元组在里面
c = np.concatenate((a,b))
print(c)

c = np.concatenate((a,b),axis=1)
print(c)

#竖着拼接
#print(np.vstack((a,b)))
#横着拼接
#print(np.hstack((a,b)))

#拉平
print(a.flatten())
# 或者 print(a.ravel())