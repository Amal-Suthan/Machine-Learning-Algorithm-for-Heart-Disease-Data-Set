# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:59:51 2020

@author: hpi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


heart = pd.read_csv('F:\local disk E\CCBT\Big Data\Assignment\input\cleaveland1.csv')
heart.head()
heart = pd.read_csv('F:\local disk E\CCBT\Big Data\Assignment\input\cleaveland1.csv', sep=",", names=["Age", "Sex", "CP", "Trestbps", "Chol", "Fbs", "Restecg", "Thalach", "Exang", "Oldpeak", "Slope", "CA", "Thal", "Label"])
heart.tail()
heart.info()
heart['Label'].unique()
heart['Label'].value_counts()

#Maplabels function will make the labels to 0 or 1
def mapLabels(value):
    if value > 0:
        return 1
    else:
        return 0
    
heart['Label'] = heart['Label'].map(mapLabels)
heart['Label'].value_counts()

#Converting the null values(?) into valid values
heart['Thal'].value_counts()
heart['CA'].value_counts()
heart['Thal'] = heart['Thal'].map(lambda x : 3 if x == '?' else int(x))
heart['Thal'].value_counts()
heart['CA'] = heart['CA'].map(lambda x : 0 if x == '?' else int(x))
heart['CA'].value_counts()
heart.info()

#Classifying the age group to child(0),young(1),Adult(2),mid-age(3) and old(4)
heart['Age'].value_counts()
heart.loc[heart['Age'] <= 16, 'Age']  = 0,
heart.loc[(heart['Age'] > 16) & (heart['Age'] <= 26), 'Age']  = 1,
heart.loc[(heart['Age'] > 26) & (heart['Age'] <= 36), 'Age']  = 2,
heart.loc[(heart['Age'] > 36) & (heart['Age'] <= 62), 'Age']  = 3,
heart.loc[heart['Age'] > 62, 'Age']  = 4
heart['Age'].value_counts()


#Classifying the blood pressure to low(0),normal(1) and high(2)
heart['Trestbps'].value_counts()
heart.loc[heart['Trestbps'] <= 100, 'Trestbps']  = 0,
heart.loc[(heart['Trestbps'] > 100) & (heart['Trestbps'] <= 125), 'Trestbps']  = 1,
heart.loc[heart['Trestbps'] > 125, 'Trestbps']  = 2
heart['Trestbps'].value_counts()

#age distibution
heart.Age.unique()
sns.distplot(heart.Age)
plt.title('Age Distribution')

#Types of chest pain vs age
result=[]
for i in heart['ChestPain']:
    if i == 1:
        result.append('Typical Angina')
    if i ==2:
        result.append('Atypical Angina')
    if i ==3:
        result.append('Non-Anginal')
    if i==4:
        result.append('Asymptomatic')
        
heart['ChestPainType']=pd.Series(result)
sns.swarmplot(x='ChestPainType', y='Age', data=heart)

#Healthy heart or diseased heart
heart_health=[]
for k in heart['Target']:
    if k == 0:
        heart_health.append('Healthy Heart')
    else:
        heart_health.append('Heart Disease')
        
#Heart Attacks in male and female

ax = sns.countplot(x='Gender',hue=heart_health,data=heart,palette='mako_r')


#Creating test and train data set
labels = heart['Label']
features = heart.drop(['Label'], axis=1)
train_x, test_x, train_y, test_y = train_test_split(features,labels, shuffle=True)



#KFold allows us to validate all over our train data and help find our best model
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#Trying Suppport Vector Machine model
clf = SVC(gamma='auto')
scoring = 'accuracy'
score = cross_val_score(clf, train_x, train_y, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)

#Trying Gradient Boosting Classifier Model
clf = GradientBoostingClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_x, train_y, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)

#Trying Decision classifier model
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_x, train_y, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)

#Tring Random Forest Classifier Model
clf = RandomForestClassifier(n_estimators=10)
scoring = 'accuracy'
score = cross_val_score(clf, train_x, train_y, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)

#Trying Naive Bayes model 
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_x, train_y, cv=k_fold, n_jobs=1, scoring=scoring)
round(np.mean(score)*100, 2)

clf = GaussianNB()
clf.fit(train_x, train_y)
predictions = clf.predict(test_x)
values = list(zip(predictions, test_y.values))
status = []
for x, y in values:
    status.append(x == y)
list_of_values = list(zip(predictions, test_y.values, status))
final_df = pd.DataFrame(list_of_values, columns=['Predicted', 'Actual', "Status"])
print(final_df)
