import pandas as pd
import numpy as np

#food_info = pd.read_csv('food_info.csv')

#对列进行排序,后面的参数表示是否对数据进行单独提取出来,默认是从小到大的排序
#补参数 ascending = False 进行降序排列
#food_info.sort_values('Sodium_(mg)', inplace=True)
#print(food_info['Sodium_(mg)'])

titanic_survival = pd.read_csv('titanic_train.csv')
#print(titanic_survival.head())

age = titanic_survival['Age']
#print(age.loc[0:10])

#提取出空值的索引位置
age_is_null = pd.isnull(age)
#print(age_is_null)
age_null_true = age[age_is_null]
#print(age_null_true)
age_null_count = len(age_null_true)
print(age_null_count)

#如果数据中包含有缺失值,那么在运算的过程中就会变成缺失值

good_ages = titanic_survival['Age'][age_is_null == False]
#平均年龄
correct_mean_age = sum(good_ages) / len(good_ages)

#当然,也可以直接调用.mean()求平均值
#correct_mean_age = titanic_survival['Age'].mean()

#如果数据中存在,我们的处理策略,不应该是把缺失数据剔除掉,可以填充上众数,中位数,平均值这些

passenger_classes = [1,2,3]
fares_by_class = {}

for this_class in passenger_classes:
    pclass_rows = titanic_survival[titanic_survival['Pclass'] == this_class]
    #定位到船票的价格
    pclass_fares = pclass_rows['Fare']
    fare_for_class = pclass_fares.mean()
    fares_by_class[this_class] = fare_for_class

#print(fares_by_class)

#用pandas自带的函数pivot_table透视表求平均
#index = 'Pclass'基准指标,    values = 'Survived' 与 index相关的关系
#aggfunc = np.mean 具体什么样的相关关系
passenger_survival = titanic_survival.pivot_table(index='Pclass', values='Survived',
                                                  aggfunc=np.mean)
#print(passenger_survival)

#求仓位区别年龄的平均值,aggfunc 默认是求均值的操作
#passenger_age = titanic_survival.pivot_table(index='Pclass', values='Age', aggfunc=np.mean)
#print(passenger_age)

#一个量与另外两个量之间的关系值
#登船地点和船票,获救与否之间的关系
port_stats = titanic_survival.pivot_table(index='Embarked', values=['Fare', 'Survived'],
                                          aggfunc=np.sum)
#print(port_stats)

#把有缺失值的列(axis=1)丢掉
drop_na_columns = titanic_survival.dropna(axis=1)

#对于列(age, sex),中有缺失值,把有缺失数据的行(axis=0)去掉
new_titanic_survival = titanic_survival.dropna(axis=0, subset=['Age', 'Sex'])

#精确定位,83行,Age列的数据
row_index_83_age = titanic_survival.loc[83, 'Age']