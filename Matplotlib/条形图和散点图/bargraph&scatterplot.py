import pandas as pd
import matplotlib.pyplot as plt
from numpy import arange

reviews = pd.read_csv('fandango_scores.csv')
cols = ['FILM', 'RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']
norm_reviews = reviews[cols]
#print(norm_reviews[:1])

num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']

#评分值
bar_heights = norm_reviews.ix[0,num_cols].values
#print(bar_heights)

#每个柱子距离0点的距离
bar_positions = arange(5) + 0.75
#print(bar_positions)

#取空图
fig,ax = plt.subplots()

#0.3:柱子的宽度
ax.bar(bar_positions,bar_heights,0.3)
#横向的条形图,要求把下面的坐标上的设置也改一下
#ax.barh(bar_positions,bar_heights,0.3)

ax.set_xticks(bar_positions)
ax.set_xticklabels(num_cols, rotation = 45)

ax.set_xlabel('Rating Source')
ax.set_ylabel('Average Rating')
ax.set_title('Average User Rating For Avengers: Age of Ultron (2015)')

plt.show()

#散点图
fig, ax = plt.subplots()
ax.scatter(norm_reviews['Fandango'])