#数据预处理中的降维,分类任务

#LDA关心的是能够最大化类间区分度的坐标轴成分
#将特征空间(数据集中的多维样本)投影到一个维度更小的k维子空间中,同时保持区分类别的信息

#监督性:LDA是'有监督'的,它计算的是另一类特定的方向
#投影:找到更合适分类的空间
#与PCA(另一种降维方法)不同,更关心分类而不是方差