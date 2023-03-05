import pandas as pd
import re

# pd.set_option('display.height', 500)
pd.set_option('display.max_rows', None)

df = pd.read_csv('train.csv')
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', None)


def p():
    print(df)


def sex_check():
    df.loc[df['Sex'] == 1,  'ratio'] = df.ratio*0.8
    df.loc[df['Sex'] == 2,  'ratio'] = df.ratio*1.2
def age

# df = holder[["PassengerId", "Survived", "Age", "Pclass", "Sex", "Fare"]]
# df.loc[df['Sex'] == 'female', 'Sex'] = "1"
# df.loc[df['Sex'] == 'male', 'Sex'] = "0"
# df.Sex.replace(['male', 'female'], [0, 1], inplace=True)

# srv = df.loc[df['Survived'] == 1]
# ded = df.loc[df['Survived'] == 0]
# srv.reset_index(drop=True, inplace=True)
# ded.reset_index(drop=True, inplace=True)

# print(df)
# df.to_csv('checkpoint.csv')
# print(df.dtypes)
# print(srv.drop('PassengerId', axis=1).mean())
# print(df.drop(['PassengerId', 'Survived'], axis=1).describe())
# print(df.drop('PassengerId', axis=1).groupby(['Survived']).mean())
#
# print(srv.drop(['PassengerId', 'Survived'], axis=1).describe())
# print(ded.drop(['PassengerId', 'Survived'], axis=1).describe())

# STARTING FRESH


df.drop(columns=['Name', 'Ticket'], inplace=True)

df['ratio'] = 1

df.loc[df['Sex'] == 'male', 'Sex'] = 1
df.loc[df['Sex'] == 'female', 'Sex'] = 2

# df['count'] = 0
# print(df.groupby(['Survived', 'Sex', 'Pclass']).count()['count'])



sex_check()
p()
