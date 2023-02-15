import pandas as pd



holder = pd.read_csv('train.csv')

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

#myst3l

# love from mac
# love from pc
# love from browser xo
