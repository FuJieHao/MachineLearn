import pandas as pd

food_info = pd.read_csv('food_info.csv')

#()中指定参数,显示前几行
#print(food_info.head())

#显示后几行  food_info.tail()
#打印出列名  food_info.columns
#形状       food_info.shape

#food_info.loc[0]   第一行数据
#food_info.loc[3:6]
#food_info.loc[2,5,10]

#food_info['NDB_No']    #根据列名取列值

#取多列数据
#columns = ['Zinc_(mg)', 'Copper_(mg)']
#zinc_copper = food_info[columns]

#将列名变成list
col_names = food_info.columns.tolist()
#print(col_names)

gram_columns = []

for c in col_names:
    if c.endswith('(g)'):
        gram_columns.append(c)

gram_df = food_info[gram_columns]
# print(gram_columns)
# print(gram_df.head(3))

#对列进行运算
#div_1000 = food_info['Iron_(mg)'] / 1000
#print(div_1000)

#对应位置进行相乘
#water_energy = food_info['Water_(g)'] * food_info['Energ_Kcal']

#print(food_info.shape)

iron_grams = food_info['Iron_(mg)'] / 1000
#新建列名,添加列
food_info['Iron_(g)'] = iron_grams

#print(food_info.shape)

#找一个列中的最值  food_info['Energ_Kcal'].(max() / min())