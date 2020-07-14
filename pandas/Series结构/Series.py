import pandas as pd
from pandas import Series

#Series相当于矩阵中的一行或一列

fandango = pd.read_csv('fandango_score_comparison.csv')
series_film = fandango['FILM']
#print(series_film)
series_rt = fandango['RottenTomatoes']
#print(series_rt)

film_names = series_film.values
rt_scores = series_rt.values
#print(type(film_name))
series_custom = Series(rt_scores, index=film_names)
print(series_custom[['Minions (2015)', 'Leviathan (2014)']])