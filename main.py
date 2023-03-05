import pandas as pd
import re

# pd.set_option('display.height', 500)
pd.set_option('display.max_rows', None)

df = pd.read_csv('train.csv')
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


def p():
    print(df)


def survival_check():
    # age_check()
    # sex_check()
    class_check()
    df['ratio'] = (df['Class S']*1)*(df['Age S']*1)


def class_check():
    df.loc[df['Pclass'] == 1, 'Class S'] = class_mean.iloc[1-1]/0.5
    df.loc[df['Pclass'] == 2, 'Class S'] = class_mean.iloc[2-1]/0.5
    df.loc[df['Pclass'] == 3, 'Class S'] = class_mean.iloc[3-1]/0.5


# def new_class_check(x):




# def sex_check():
#     df.loc[df['Sex'] == 1,  'ratio'] = df.ratio*0.8
#     df.loc[df['Sex'] == 2,  'ratio'] = df.ratio*1.2
#
#
# def age_check():
#     df.loc[df['Age'] > 28,  'ratio'] = df.ratio*0.8
#     df.loc[df['Sex'] <= 28,  'ratio'] = df.ratio*1.2


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


# df.loc[df['Sex'] == 'male', 'Sex'] = 1
# df.loc[df['Sex'] == 'female', 'Sex'] = 2
# STARTING FRESH
# print(df.dtypes)

df.drop(columns=['Name', 'Ticket'], inplace=True)

df['Class S'] = 1
df['Sex S'] = 1
df['Age S'] = 1
df['Sib S'] = 1
df['Parch S'] = 1
df['Fare S'] = 1
df['Embarked S'] = 1
df['ratio'] = 1

class_mean = df.groupby('Pclass').mean(numeric_only=True)['Survived']
sex_mean = df.groupby('Sex').mean(numeric_only=True)['Survived']



# df['count'] = 0
# print(df.groupby(['Survived', 'Sex']).count()['count'])
#
#
# print(df.groupby('Sex').mean(numeric_only=True)['Survived'])

print(sex_mean.describe)



survival_check()

# print(df)