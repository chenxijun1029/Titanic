import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve


data_train = pd.read_csv("train.csv")

data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#数据预处理
#Cabin属性缺失值较多，按照"有"或"无"处理
data_train.loc[ (data_train.Cabin.notnull()), 'Cabin' ] = "Yes" 
data_train.loc[ (data_train.Cabin.isnull()), 'Cabin' ] = 'No'

#Age属性为连续属性，缺失值较少，可以用平均值填充（之后分成不同年龄段）
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].mean())

#Embarked属性缺失值较少，可以剔除掉
data_train.dropna(axis=0, inplace=True)

#Ticket,Name和PassengerId属性没有太多信息，可以剔除掉
data_train.drop(['Ticket','Name','PassengerId'], axis=1, inplace=True)

#对属性值类型为字符串的列(Cabin,Sex和Embarked)先转换成字典形式，使用独热编码
vec = DictVectorizer(sparse=False, dtype=np.int8)
data_train = vec.fit_transform(data_train.to_dict(orient='record'))
data_train = pd.DataFrame(data_train, columns=vec.feature_names_)

#Age和Fare属性需要scaling,否则使用一些以距离为误差函数的算法（如逻辑回归，线性回归）时，收敛太慢甚至不收敛
age_scaling = StandardScaler().fit(data_train['Age'])
data_train['Age'] = StandardScaler().fit_transform(data_train['Age'], age_scaling)
fare_scaling = StandardScaler().fit(data_train['Fare'])
data_train['Fare'] = StandardScaler().fit_transform(data_train['Fare'], fare_scaling)

#将数据分为输入x和输出y,并且留出验证集
y = data_train['Survived']
x = data_train.drop('Survived', axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 1)

#逻辑回归建模
lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
#留出验证
lr.fit(xtrain, ytrain)
y_lr = lr.predict(xtest)
accuracy_score(ytest, y_lr)
#交叉验证
cross_validation.cross_val_score(lr, x, y, cv=5)

#bagging
lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_lr = BaggingRegressor(lr, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1)
bagging_lr.fit(xtrain, ytrain)
y_lr = bagging_lr.predict(xtest).astype(np.int32)
accuracy_score(ytest, y_lr)

#绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(bagging_lr, x, y, cv=7, n_jobs=1, train_sizes=np.linspace(.05, 1., 20), verbose=0)
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color="b", label="scores of train")
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', color="r", label="scores of test")
plt.legend()
plt.xlabel('samples')
plt.ylabel('scores')

#对test数据集采取同样的数据处理
data_test = pd.read_csv('test.csv')

data_test.loc[ (data_test.Cabin.notnull()), 'Cabin' ] = "Yes" 
data_test.loc[ (data_test.Cabin.isnull()), 'Cabin' ] = 'No'
data_test['Fare'].fillna(data_test['Fare'].mean(), inplace =True)
data_test['Age'].fillna(data_test['Age'].mean(), inplace =True)
data_test.drop(['Ticket','Name','PassengerId'], axis=1, inplace=True)
vec = DictVectorizer(sparse=False, dtype=np.int8)
data_test = vec.fit_transform(data_test.to_dict(orient='record'))
data_test = pd.DataFrame(data_test, columns=vec.feature_names_)
age_scaling = StandardScaler().fit(data_test['Age'])
data_test['Age'] = StandardScaler().fit_transform(data_test['Age'], age_scaling)
fare_scaling = StandardScaler().fit(data_test['Fare'])
data_test['Fare'] = StandardScaler().fit_transform(data_test['Fare'], fare_scaling)

#生成结果文件
predictions = bagging_lr.predict(data_test)
result = pd.DataFrame({'PassengerId':pd.read_csv('test.csv')['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_bagging_predictions.csv", index=False)






