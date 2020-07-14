import numpy as np

#指定类型
fj_array = np.array([1,2,3,4,5],dtype=np.float32)

#占用的字节
print(fj_array.nbytes)

#使数组有任何形式
fj_array = np.array([1,10,3.5,'str'],dtype=np.object)
print(fj_array)


fj_array = np.array([1,2,3,4,5])
print(np.asarray(fj_array,dtype=np.float32)) #不会改变原始的内容
print(fj_array)

print(fj_array.astype(np.float32)) #同样不会改变原始的内容