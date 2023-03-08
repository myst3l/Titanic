# Data wrangling
import pandas as pd
import numpy as np
import missingno
# from collections import Counter
#
# # Data visualisation
import seaborn as sns
import matplotlib.pyplot as plt
#
# # Machine learning models
# from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# # from catboost import CatBoostClassifier
#
# # Model evaluation
# from sklearn.model_selection import cross_val_score
#
# # Hyperparameter tuning
# from sklearn.model_selection import GridSearchCV

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# STARTING HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('train.csv')


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

