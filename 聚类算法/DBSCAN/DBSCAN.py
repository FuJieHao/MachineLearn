#核心对象:若某个点的密度达到算法设定的阈值则其为核心点
#领域的距离阈值:设定的半径r
#直接密度可达:若某点p在q的r领域内,且q是核心点则p-q直接密度可达.
#密度可达:若有一个点的序列q0.q1...qk,对任意qi - qi-1 是直接密度可达的,则称从q0到qk密度可达,这实际上是直接密度可达的'传播'

#参数:
#数据集
#半径r
#密度阈值

#优势:
#不需要指定簇个数
#可以发现任意形状的簇
#擅长找到离群点
#两个参数就够了


#劣势:
#高维数据有些困难(可以做降维)
#参数难以选择(参数对结果的影响非常大)
#Sklearn中效率很慢(数据削减策略)