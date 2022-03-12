import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
print(wine.data, wine.target)
pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,
                                                wine.target,
                                                test_size=0.3,
                                                random_state=420)
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
feature_name = [
    '酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调',
    'od280/od315稀释葡萄酒', '脯氨酸'
]

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)  # 返回预测的准确度
print(score)

dot_data = tree.export_graphviz(clf,
                                out_file='tree.dot',
                                feature_names=feature_name,
                                class_names=["琴酒", "雪莉", "贝尔摩德"],
                                filled=True,
                                rounded=True)
