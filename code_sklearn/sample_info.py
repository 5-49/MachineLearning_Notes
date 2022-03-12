from sklearn.datasets import load_wine

wine = load_wine()
print(wine.keys())  # 查看键(属性)
print(wine.data.shape, wine.target.shape)
# 查看数据的形状 (178, 13) (178,) 行和列 178个样本，每个13列,target是长178的一维数组
# data是特征数据集，target是分类目标结果
print(wine.feature_names)  # 查看有哪些特征
print(wine.target_names)  # 查看有哪些特征
print(wine.DESCR)  # described 描述这个数据集的信息
