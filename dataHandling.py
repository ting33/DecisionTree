import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
train_data=pd.read_csv("/Users/zhouya/Documents/code/Titanic_Data/train.csv")
test_data=pd.read_csv("/Users/zhouya/Documents/code/Titanic_Data/test.csv")
# print(train_data.info())
print(test_data.head())
print('-'*40)
print(train_data.head())
# print('-'*40)
# print(train_data.describe())
# print('-'*40)
# print(train_data.describe(include=['O']))
# print('-'*40)
# print(train_data.tail())
# print(train_data.shape)
#是否数据是否有缺失
# print(train_data.isnull().any(axis = 0))
# print(train_data.isnull())
# print(train_data.isnull().sum(axis=0))
#用品平均数填充
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
#查看有多少种不同的值
print(train_data['Embarked'].value_counts())
print(test_data['Embarked'].value_counts())
# 使用登录最多的港口来填充登录港口的nan值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# print(train_data.isnull().sum(axis=0)/train_data.shape[0])
# print(test_data.isnull().sum(axis=0)/test_data.shape[0])
# print('数据集是否存在重复观测: \n',any(train_data.duplicated()))

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))