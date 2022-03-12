from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 1.读取数据集
wine = load_wine()

# 2.划分数据与标签
wine.data = wine.data[:, 0:2]  # 为便于后边画图显示，只选取前两维度。
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,
                                                wine.target,
                                                test_size=0.2,
                                                random_state=1)

# 3.训练svm分类器
classifier = svm.SVC(C=2,
                     kernel='rbf',
                     gamma=10,
                     decision_function_shape='ovr')  # ovr:一对多策略
classifier.fit(Xtrain, Ytrain.ravel())  # ravel函数在降维时默认是行序优先

# 4.计算svc分类器的准确率
print("训练集：", classifier.score(Xtrain, Ytrain))
print("测试集：", classifier.score(Xtest, Ytest))

# 4.计算svc分类器的准确率
print("训练集：", classifier.score(Xtrain, Ytrain))
print("测试集：", classifier.score(Xtest, Ytest))

# 查看决策函数
print('train_decision_function:',
      classifier.decision_function(Xtrain))
print('predict_result:', classifier.predict(Xtrain))

# 5.绘制图形
x = wine.data[:, 0:2]
y = wine.target
print(x.shape)  # 178,2
print(y.shape)  # 178,
# 确定坐标轴范围
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 设置颜色
cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])

grid_hat = classifier.predict(grid_test)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=cm_dark)  # 样本
plt.scatter(Xtest[:, 0],
            Xtest[:, 1],
            c=Ytest,
            s=30,
            edgecolors='k',
            zorder=2,
            cmap=cm_dark)  # 圈中测试集样本点
plt.xlabel('特征一', fontsize=13)
plt.ylabel('特征二', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('红酒SVM二特征分类')
plt.show()
