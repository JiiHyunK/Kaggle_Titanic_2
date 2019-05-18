import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.width',3200)
pd.set_option('display.max_columns',1000)

train = pd.read_csv('~/Kaggle/titanic_2/data/train.csv')
test = pd.read_csv('~/Kaggle/titanic_2/data/test.csv')
test_cpy=test.copy()

#1.Analyze by pivoting features
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False),'\n')
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False),'\n')
print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False),'\n')
print(train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False),'\n')

#2.visualization
g = sns.FacetGrid(train, col='Survived')
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

combine = pd.concat([train, test], sort=False)


#3. data manage
def clean_data_makeTitle():
    #dropping features
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

    title_into_ordinal={'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Rare':4}
    combine['Title']=combine['Title'].map(title_into_ordinal)
    combine.drop(['Name','PassengerId','Ticket','Cabin'],axis=1,inplace=True)

def clean_data_Sex():
    #make ordinal
    combine['Sex']=combine['Sex'].map({'female':0,'male':1})
def clean_data_Embarked():
    #fill null data of 'Embarked' & make ordinal
    combine['Embarked']=combine['Embarked'].fillna('S')     #2개 missing된거 simply fill with most frequent
    combine['Embarked']=combine['Embarked'].map({'C':0,'Q':1,'S':2})

def clean_data_FamilySize(combine):
    combine['FamilySize']=combine['Parch']+combine['SibSp']+1

    combine['IsAlone']=0
    combine.loc[combine['FamilySize']==1,'IsAlone']=1
    combine=combine.drop(['SibSp','Parch'],axis=1)
    print(combine.info())
    
def clean_data_Fare():
    combine['Fare'].fillna(combine['Fare'].median(),inplace=True)
    mean=combine[['FamilySize','Fare']].groupby('FamilySize',as_index=False).mean()
    combine['FareBand']=pd.cut(combine['Fare'],4)
    print(combine['FareBand'].unique())
    combine.loc[combine['Fare']<=128,['Fare']]=0
    combine.loc[(combine['Fare']>7.9)&(combine['Fare']<=14.5),['Fare']]=1
    combine.loc[(combine['Fare']>14.5)&(combine['Fare']<=31.3),['Fare']]=2
    combine.loc[combine['Fare']>31.3,['Fare']]=3
    combine.drop('FareBand',axis=1,inplace=True)
    
def clean_data_Age():
    guess_ages = np.zeros((5,3))
    for i in range(0, 5):
        for j in range(0, 3):
            guess_df = combine[(combine['Title'] == i) & (combine['Pclass'] == j + 1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()
            if ((i == 4) & (j == 2)):
                age_guess = 0
            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 5):
        for j in range(0, 3):
            combine.loc[(combine.Age.isnull()) & (combine.Title == i) & (combine.Pclass == j + 1),'Age'] = guess_ages[i, j]
    
    combine['AgeBand']=pd.cut(combine['Age'],5)
    combine.loc[ combine['Age'] <= 16, 'Age'] = 0
    combine.loc[(combine['Age'] > 16) & (combine['Age'] <= 32), 'Age'] = 1
    combine.loc[(combine['Age'] > 32) & (combine['Age'] <= 48), 'Age'] = 2
    combine.loc[(combine['Age'] > 48) & (combine['Age'] <= 64), 'Age'] = 3
    combine.loc[ combine['Age'] > 64, 'Age']=4
    combine.drop('AgeBand',axis=1,inplace=True)


clean_data_makeTitle()
clean_data_Sex()
clean_data_Embarked()
clean_data_FamilySize(combine)
clean_data_Fare()
clean_data_Age()

#Modeling
train=combine[:891]
test=combine[891:combine.shape[0]]
test.drop('Survived',axis=1,inplace=True)

X_train=train.drop('Survived',axis=1)
Y_train=train['Survived']
X_test=test

from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

#<Logistic Regression>
logreg= LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred=logreg.predict(X_test)
acc_logreg=round(logreg.score(X_train,Y_train)*100,2)
print(acc_logreg)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_svc)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)
Y_pred_ans=Y_pred
Y_pred_ans=Y_pred_ans.astype(int)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_logreg,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))


submission = pd.DataFrame({
        "PassengerId": test_cpy['PassengerId'],
        "Survived": Y_pred_ans
    })
submission.to_csv('~/Kaggle/titanic_2/submission.csv', index=False)
