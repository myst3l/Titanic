import pandas as pd
import re

# pd.set_option('display.height', 500)
pd.set_option('display.max_rows', None)

df = pd.read_csv('train.csv')
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


def survival_check():
    # age_check()
    sex_check()
    class_check()
    sib_check()
    df['Ratio'] = (df['Class S']*1)*(df['Sib S']*1)*(df['Sex S']*1)

    df.loc[((df['Survived'] == 1) & (df['Ratio'] >= 1)), 'Alive Correct'] = 1
    df.loc[((df['Survived'] == 0) & (df['Ratio'] <= 1)), 'Dead Correct'] = 1
    alive_c = sum(df['Alive Correct'])
    dead_c = sum(df['Dead Correct'])
    print(alive_c, "/342 for alive are correct. Accuracy is: ", (alive_c/342)*100, "%")
    print(dead_c, "/549 for dead are correct. Accuracy is: ", (dead_c/549)*100, "%")
    print(dead_c+alive_c, "/891 for Total are correct. Accuracy is: ", ((dead_c+alive_c)/891)*100, "%")


def class_check():
    for x in range(1, 4):
        df.loc[df['Pclass'] == x, 'Class S'] = class_mean[x] / 0.5
    # df.loc[df['Pclass'] == 1, 'Class S'] = class_mean.iloc[1-1]/0.5
    # df.loc[df['Pclass'] == 2, 'Class S'] = class_mean.iloc[2-1]/0.5
    # df.loc[df['Pclass'] == 3, 'Class S'] = class_mean.iloc[3-1]/0.5


def sib_check():
    for x in range(6):
        df.loc[df['SibSp'] == x, 'Sib S'] = sib_mean[x] / 0.5

    df.loc[df['SibSp'] == 8, 'Sib S'] = sib_mean[8] / 0.5


def sex_check():
    df.loc[df['Sex'] == 'male', 'Sex S'] = sex_mean['male'] / 0.5
    df.loc[df['Sex'] == 'female', 'Sex S'] = sex_mean['female'] / 0.5
#
#
# def age_check():
#     df.loc[df['Age'] > 28,  'Ratio'] = df.Ratio*0.8
#     df.loc[df['Sex'] <= 28,  'Ratio'] = df.Ratio*1.2

# RANDOM STUFF FROM BEFORE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# print(df)
# df.to_csv('checkpoint.csv')
# print(df.dtypes)
# print(srv.drop('PassengerId', axis=1).mean())
# print(df.drop(['PassengerId', 'Survived'], axis=1).describe())
# print(df.drop('PassengerId', axis=1).groupby(['Survived']).mean())
#
# srv = df.loc[df['Survived'] == 1]
# ded = df.loc[df['Survived'] == 0]
# srv.reset_index(drop=True, inplace=True)
# ded.reset_index(drop=True, inplace=True)

# print(srv.drop(['PassengerId', 'Survived'], axis=1).describe())
# print(ded.drop(['PassengerId', 'Survived'], axis=1).describe())


# df.loc[df['Sex'] == 'male', 'Sex'] = 1
# df.loc[df['Sex'] == 'female', 'Sex'] = 2

# srv = df.loc[df['Ratio'] >= 1]
# srv = df.loc[df['Survived'] == 1]
# srv.reset_index(drop=True, inplace=True)
# print(srv)
# STARTING FRESH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# print(df.dtypes)

df.drop(columns=['Name', 'Ticket'], inplace=True)

df['Class S'] = 1
df['Sex S'] = 1
df['Age S'] = 1
df['Sib S'] = 1
df['Parch S'] = 1
df['Fare S'] = 1
df['Embarked S'] = 1
df['Ratio'] = 1
df['Alive Correct'] = 0
df['Dead Correct'] = 0

class_mean = df.groupby('Pclass').mean(numeric_only=True)['Survived']
sex_mean = df.groupby('Sex').mean(numeric_only=True)['Survived']
age_mean = df.groupby('Age').mean(numeric_only=True)['Survived']
sib_mean = df.groupby('SibSp').mean(numeric_only=True)['Survived']
#

# df['count'] = 0
# print(df.groupby(['Survived', 'Sex']).count()['count'])
#
#
# print(df.groupby('Sex').mean(numeric_only=True)['Survived'])

# print(sex_mean.describe)

# print(class_mean)
# print(sib_mean[8])

survival_check()



