import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

unrate = pd.read_csv('UNRATE.csv')

#print(unrate.head(12))

#first_twelve = unrate[0:12]
#x,y轴
#plt.plot(first_twelve['DATE'],first_twelve['VALUE'])
#指定角度
#plt.xticks(rotation = 45)
#加x月份
#plt.xlabel('Month')
#加y失业率
#plt.ylabel('失业率')
#plt.title('1948年失业率')
#展示
#plt.show()

#fig = plt.figure()



unrate['MONTH'] = pd.DatetimeIndex(unrate['DATE']).month
# unrate['MONTH'] = pd.DatetimeIndex(unrate['DATE']).month
#
# #unrate['MONTH'] = unrate['DATE'].dt.month
# #unrate['MONTH'] = unrate['DATE'].dt.month
# #(6,6)宽 高
# fig = plt.figure(figsize=(6,6))
# #fig = plt.figure()
# ax1 = fig.add_subplot(3,2,1)
# ax2 = fig.add_subplot(3,2,2)
# ax3 = fig.add_subplot(3,2,3)
#
# ax1.plot(np.random.randint(1,5,5), np.arange(5))
# ax2.plot(np.arange(10) * 3,np.arange(10))
#
# ax3.plot(unrate[0:12]['MONTH'],unrate[0:12]['VALUE'],c = 'red')
# ax3.plot(unrate[12:24]['MONTH'],unrate[12:24]['VALUE'],c = 'blue')
#
# plt.show()


fig = plt.figure(figsize=(10,6))
colors = ['red','blue','green','orange','black']
for i in range(5):

    start_index = i * 12
    end_index = (i + 1) * 12
    subset = unrate[start_index:end_index]
    label = str(1948 + i)
    plt.plot(subset['MONTH'],subset['VALUE'], c = colors[i], label = label)

#表示框的位置
            # 'best'            0
            # 'upper right'     1
            # 'upper left'      2
            # 'lower left'      3
            # 'lower right'     4
            # 'right'           5
            # 'center left'     6
            # 'center right'    7
            # 'lower center'    8
            # 'upper center'    9
            # 'center'          10
#print(help(plt.legend))
plt.legend(loc = 'best')
plt.xlabel('Month,Integer')
plt.ylabel('Unemployment Rate, Percent')
plt.title('Monthly Unemployment Trends, 1948-1952')

plt.show()