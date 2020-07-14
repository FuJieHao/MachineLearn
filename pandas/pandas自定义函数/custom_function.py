import pandas as pd
import numpy as np

titanic_survival = pd.read_csv('titanic_train.csv')

#对数据进行按规则排序后,同时更改其索引

#对数据进行按年龄排序
new_titanic_survival = titanic_survival.sort_values('Age',ascending=False)
#print(new_titanic_survival[0:10])

#重新设置索引
titanic_reindexed = new_titanic_survival.reset_index(drop=True)
#print(titanic_reindexed[0:10])


#pandas的自定义函数块
def hundredth_row(column):
    hundredth_item = column.loc[99]
    return hundredth_item

#这里的参数传递进去函数的名字
hundredth_row = titanic_survival.apply(hundredth_row)
print(hundredth_row)