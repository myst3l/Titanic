# Data wrangling
import pandas as pd
import numpy as np
import re
# import missingno
# from collections import Counter

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

combine = pd.concat([df, test], axis=0).reset_index(drop=True)
combine.drop(['Ticket'], axis=1, inplace=True)

combine['Sex'] = combine['Sex'].map({'male': 0, 'female': 1})  # Converting str to int


combine['Cabin'].fillna(value='U', inplace=True)     # Filling empty values

# print(combine.head())

# print(combine.isnull().sum().sort_values(ascending=False))

# print(combine[combine['Embarked'].isnull()].index)
# print(list(combine[combine['Embarked'].isnull()].index))

# empty = list(combine[combine['Age'].isnull()].index)

# print(combine['Fare'].median())
combine['Embarked'].fillna(value=combine['Embarked'].mode()[0], inplace=True)       # Filling empty values
combine['Fare'].fillna(value=combine['Fare'].median(), inplace=True)
#
# for index in empty:
#     print(combine['Age'].iloc[index])

# print(combine.isnull().sum().sort_values(ascending=False))
# print(combine.loc[combine.Cabin.str.contains('C')].index)


# for index in list(combine.loc[combine.Cabin.str.contains('C')].index):
#     combine['Cabin'].iloc[index] = 'C'

# print(combine.loc[combine.Cabin.str.contains('C')]['Cabin'])


# combine.loc[combine.Cabin.str.contains('C')]['Cabin'] = 'C'

for index in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    combine.loc[combine.Cabin.str.contains(index), 'Cabin'] = index            # Removing letters from Cabins


