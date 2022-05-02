import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

year_list = [11, 12, 13, 14, 15, 16, 17, 18, 20]

for i in year_list:
    i = str(i)
    x = pd.read_csv(
        '.\\data\\20' + i + '.csv',
        usecols=[
            'assets',
            'liabilities',
            'buffer',
            'weights',
            'impaired_loans_divide_by_gross_customer_loans_advances',
            'loan_loss_reserves_divide_by_impaired_loans',
            'customer_loans_advances_divide_by_total_assets',
            'net_charge_offs_divide_by_average_gross_customer_loans_advances',
            'unreserved_impaired_loans_divide_by_equity'
        ]).values
    y = pd.read_csv('.\\data\\20' + i + '.csv',
                    usecols=['credit rank']).values.ravel()

    # print(x.shape, y.shape)  # 14245,4  14245,
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    feature_name = ['assets', 'liabilities', 'buffer', 'weights']

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest, Ytest)  # 返回预测的准确度
    print('(9 factors)' + '20' + i + 'Decision Tree Score: %.4f' % score)

# 画图
# dot_data = tree.export_graphviz(clf,
#                                 out_file='res_tree 2011.dot',
#                                 feature_names=feature_name,
#                                 filled=True,
#                                 rounded=True)
