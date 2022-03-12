from sklearn import naive_bayes
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
print(wine.data, wine.target)
print(wine.data.shape, wine.target.shape)  # 178,13  178
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,
                                                wine.target,
                                                test_size=0.2,
                                                random_state=0,
                                                stratify=wine.target)

cls = naive_bayes.GaussianNB()
cls.fit(Xtrain, Ytrain)
# print('GaussianNB Training Score: %.2f' % cls.score(X_train, y_train))
print('GaussianNB Testing Score: %.2f' % cls.score(Xtest, Ytest))

cls = naive_bayes.MultinomialNB()
cls.fit(Xtrain, Ytrain)
# print('MultinomialNB Training Score: %.2f' % cls.score(X_train, y_train))
print('MultinomialNB Testing Score: %.2f' % cls.score(Xtest, Ytest))

cls = naive_bayes.BernoulliNB()
cls.fit(Xtrain, Ytrain)
# print('BernoulliNB Training Score: %.2f' % cls.score(X_train, y_train))
print('BernoulliNB Testing Score: %.2f' % cls.score(Xtest, Ytest))
