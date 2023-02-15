import pandas as pd


class Passenger:
    def __init__(self, pid, survived, social, sex, age, fare):
        self.pid = pid
        self.survived = survived
        self.social = social
        self.sex = sex
        self.age = age
        self.fare = fare


holder = pd.read_csv('C:/Users/Kumayl/Desktop/titanic/train.csv')

df = holder[["PassengerId", "Survived", "Pclass", "Sex", "Fare"]]
df.loc[df['Sex'] == 'female', 'Sex'] = 2
df.loc[df['Sex'] == 'male', 'Sex'] = 1

srv = df.loc[df['Survived'] == 1]
ded = df.loc[df['Survived'] == 0]
srv.reset_index(drop=True, inplace=True)
ded.reset_index(drop=True, inplace=True)
print(srv)
print(ded)

srv.to_csv('survivors.csv')
