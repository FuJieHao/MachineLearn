import numpy as np

#
#data = []
#with open('fj.txt') as f:
#    for line in f.readlines():
#        fileds = line.split()
#        cur_data = [float(x) for x in fileds]
#        data.append(cur_data)

#data = np.array(data)
#print(data)


#',' 指定分隔符 默认是空格 不处理第一行
# 'fj.txt' 路径
# skiprows 去掉第几行
# delimiter 指定分隔符
# usecols = (0,1,4) 指定使用哪几列
data = np.loadtxt('fj.txt',delimiter=' ',skiprows=1)
print(data)