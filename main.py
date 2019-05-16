import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width',3200)
pd.set_option('display.max_columns',1000)


train = pd.read_csv('~/Kaggle/titanic_2/data/train.csv')
test = pd.read_csv('~/Kaggle/titanic_2/data/test.csv')

print(train.nunique())
print(train.describe(include='O'))

#3. Analyze by pivoting features
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False),'\n')
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False),'\n')
print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False),'\n')
print(train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False),'\n')



#4. data visualization
g = sns.FacetGrid(train, col='Survived') #survived경우의수(0,1)만을 가지고 Age를 나타내자. 근까 죽었을때의 age histogram 안죽엇을때의 age histogram
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

grid = sns.FacetGrid(train, col='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette='deep')
grid.add_legend()

grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

plt.show()

#5. data manage
combine=pd.concat([train,test],sort=False)
combine['Title']=combine['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
print(pd.crosstab(combine['Title'],combine['Sex']))


def Title_mapping():
    dict={'Mr':'Mr',
          'Mrs':'Mrs',
          'Miss':'Miss',
          'Master':'Master',
          'Don':'Rare',
          'Rev':'Rare',
          'Dr':'Rare',
          'Mme':'Mrs',
          'Ms':'Miss',
          'Major':'Rare',
          'Lady':'Rare',
          'Sir':'Rare',
          'Mlle':'Rare',
          'Col':'Rare',
          'Capt':'Rare',
          'the Countess':'Rare',
          'Jonkheer':'Rare',
          'Dona':'Rare'}
    combine['Title']=combine['Title'].map(dict)
Title_mapping()

print(combine[['Title','Survived']].groupby(['Title'],as_index=False).mean())

title_into_ordinal={'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Rare':4}
combine['Title']=combine['Title'].map(title_into_ordinal)
combine.drop(['Name','PassengerId','Ticket','Cabin'],axis=1,inplace=True)

#make ordinal
combine['Sex']=combine['Sex'].map({'female':0,'male':1})
#fill null data of 'Embarked' & make ordinal
combine['Embarked']=combine['Embarked'].fillna('S')     #2개 missing된거 simply fill with most frequent
combine['Embarked']=combine['Embarked'].map({'C':0,'Q':1,'S':2})

#fill null data of 'Age'
grid= sns.FacetGrid(combine,row='Pclass',col='Title')
grid.map(plt.hist,'Age')
grid.add_legend()
plt.show()

corr_matrix=combine.corr()
print(corr_matrix['Age'].sort_values(ascending=True))
guess_ages = np.zeros((3,5))

for i in range(0, 3):
    for j in range(0, 5):
        guess_df = combine[(combine['Pclass'] == i+1) & (combine['Title'] == j)]['Age'].dropna()
        age_guess = guess_df.median()
        # print(age_guess)
        # Convert random age float to nearest .5 age
        if ((i == 2) & (j == 4)):
            age_guess = 0
        guess_ages[i, j] = age_guess

for i in range(0, 3):
    for j in range(0, 5):
        combine.loc[(combine.Age.isnull()) & (combine.Pclass == i+1) & (combine.Title == j), 'Age'] = guess_ages[i, j]


combine['AgeBand']=pd.cut(combine['Age'],5)
print(combine[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean())

#make 'AgeBand'
combine.loc[combine['Age']<16,'Age']=0
combine.loc[(combine['Age']>=16)&(combine['Age']<32),'Age']=1
combine.loc[(combine['Age']>=32)&(combine['Age']<48),'Age']=2
combine.loc[(combine['Age']>=48)&(combine['Age']<64),'Age']=3
combine.loc[combine['Age']>=64,'Age']=4

combine.drop(['AgeBand'],axis=1,inplace=True)



#make 'FamilySize'
combine['FamilySize']=combine['Parch']+combine['SibSp']+1

#make 'IsAlone'
combine['IsAlone']=0
combine.loc[combine['FamilySize']==1,'IsAlone']=1
print(combine)

