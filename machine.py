# Data wrangling
import pandas as pd
import numpy as np
# import missingno
# from collections import Counter
#
# # Data visualisation
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Machine learning models
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# from catboost import CatBoostClassifier
#
# # Model evaluation
from sklearn.model_selection import cross_val_score
#
# # Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# STARTING HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# print(df.info())
#
# print(df.isnull().sum().sort_values(ascending=False))
# print(df.shape)

# print(df['Sex'].value_counts(dropna=False))
# print(df[['Sex', 'Survived']].groupby('Sex', as_index = False).mean().sort_values(by = 'Survived', ascending = False))

# print(df['Pclass'].value_counts(dropna=False))
# print(df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean().sort_values(by='Survived', ascending=False))

# print(df['Embarked'].value_counts(dropna=False))
# print(df[['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean().sort_values(by='Survived', ascending=False))

# print(df['SibSp'].value_counts(dropna=False))
# print(df[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean().sort_values(by='Survived', ascending=False))

# print(df['Parch'].value_counts(dropna=False))
# print(df[['Parch', 'Survived']].groupby('Parch', as_index=False).mean().sort_values(by='Survived', ascending=False))


# print(df['Fare'].value_counts(dropna=False))
# print(df[['Fare', 'Survived']].groupby('Fare', as_index=False).mean().sort_values(by='Survived', ascending=False))

df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)



# print(df['Embarked'].value_counts(dropna=False))     # Finding and filing NA values for Embarked in train dataset
embarked_mode = df['Embarked'].dropna().mode()[0]
# print(embarked_mode)
df['Embarked'].fillna(embarked_mode, inplace=True)


median_fare = test['Fare'].dropna().median()           # Finding and filing NA values for Fare in Test dataset
# print(median_fare)
test['Fare'].fillna(median_fare, inplace=True)

test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})     # mapping male and female to numbers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

combine = pd.concat([df, test], axis=0).reset_index(drop=True)  # combining both data sets

for col in ['Embarked']:                           # Other way to remap
    combine[col] = le.fit_transform(combine[col])
    # print(le.classes_)

# print(combine.isnull().sum().sort_values(ascending=False))


age_nan_indices = list(combine[combine['Age'].isnull()].index)

# print(age_nan_indices)
# print(len(age_nan_indices))

# Loop through list and impute missing ages

for index in age_nan_indices:
    median_age = combine['Age'].median()
    predict_age = combine['Age'][(combine['SibSp'] == combine.iloc[index]['SibSp'])
                                 & (combine['Parch'] == combine.iloc[index]['Parch'])
                                 & (combine['Pclass'] == combine.iloc[index]["Pclass"])].median()
    if np.isnan(predict_age):
        combine['Age'].iloc[index] = median_age
    else:
        combine['Age'].iloc[index] = predict_age


# print(combine['Age'].isnull().sum())

combine['Fare'] = combine['Fare'].map(lambda x: np.log(x) if x > 0 else 0)  # reducing skewness

# NEED SOME MORE PROCESSING BEFORE SEPARATING!!!!!!!!!!!

# Separate training and test set from the combined dataframe

df = combine[:len(df)]
test = combine[len(df):]

# Drop PassengerId from data
df.drop('PassengerId', axis=1, inplace=True)


# Convert survived back to integer in the training set

df['Survived'] = df['Survived'].astype('int')
# print(df.head())

test = test.drop('Survived', axis=1)
# print(test.head())

# MODELLING SOME WEIRD STUFF !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
X_train = df.drop('Survived', axis=1)
Y_train = df['Survived']
X_test = test.drop('PassengerId', axis=1).copy()
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("X_test shape: ", X_test.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)
print(len(Y_pred))