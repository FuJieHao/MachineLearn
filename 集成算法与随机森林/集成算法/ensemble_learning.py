#目的:让机器学习效果更好

#bagging:训练多个分类器取平均  (并行的进行训练,进行平均化)
#最典型的代表就是随机森林
#随机: 数据 采样随机, 特征 选择随机
#森林:很多个决策树并行放在一起

#随机森林的优势:
#它能够处理很高维度(feature)的数据,并且不用做特征选择
#在训练完成以后,它能够给出哪些feature比较重要
#容易做成并行化方法,速度比较快
#可以进行可视化展示,便于分析



#boosting: 从弱学习器开始加强,通过加权来进行训练
#Fm(x) = Fm-1(x)[前面的树] + [当前的树] (串行)
#代表模型:AdaBoost     Xgboost

#AdaBoost 会根据前一次的分类效果调整数据权重
#如果某一个数据在这次分错了,那么在下一次就会给它更大的权重
#最终的结果:每个分类器会根据自身的准确性来确定各自的权重,再合体


#Stacking:聚合多个分类或回归模型(可以分阶段来做)
#堆叠:很暴力,使用各种分类器
#可以堆叠各种各样的分类器(KNN,SVM,RF)
#分阶段:第一阶段得出各自结果,第二阶段再用前一阶段结果训练