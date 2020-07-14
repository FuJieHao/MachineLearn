import numpy as np

fj_array = np.array([[1,2,3],[4,5,6]])

#写入文件 ,以整数形式保存,以逗号为分隔符
np.savetxt('fj.txt',fj_array,fmt='%d',delimiter=',')

#读写array结构
fj_array = np.array([[1,2,3],[4,5,6]])
np.save('fj_array.npy',fj_array)

fj = np.load('fj_array.npy')
print(fj)

fj2 = np.arange(10)
np.savez('fj.npz',a = fj_array, b = fj2)

data = np.load('fj.npz')
print(data.keys())

print(data['a'])