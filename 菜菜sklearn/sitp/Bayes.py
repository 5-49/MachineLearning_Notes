import pandas as pd
from sklearn import naive_bayes
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
    # 这两种贝叶斯精确度没有第三种高
    # clf = naive_bayes.GaussianNB()
    # clf.fit(Xtrain, Ytrain)
    # print('GaussianNB Testing Score: %.2f' % clf.score(Xtest, Ytest))
    # clf = naive_bayes.MultinomialNB()
    # clf.fit(Xtrain, Ytrain)
    # print('MultinomialNB Testing Score: %.2f' % clf.score(Xtest, Ytest))

    clf = naive_bayes.BernoulliNB()
    clf.fit(Xtrain, Ytrain)
    print('(9 factors)' + '20' + i + 'Naive Bayes Score: %.4f' % clf.score(Xtest, Ytest))
