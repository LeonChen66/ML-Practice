# -*- coding: utf-8 -*-
"""
Created on Tue May 23 00:34:55 2017

@author: Leon
"""
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

df_wine.columns =['Class label','Alcohol','Malic acid','Ash','Alcalinity of Ash'\
                  ,'Magnesium','Total phenols','flavanoids','Nonflavanoid phenols'
                  ,'Proanthocyanins','Color intensity','Hue','OD280/315 of diluted winess'\
                  ,'Proline']
print('Class labels',np.unique(df_wine['Class label']))

X, y =df_wine.iloc[:,1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std  =stdsc.transform(X_test)

lr = LogisticRegression(penalty = 'l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy: ',lr.score(X_train_std,y_train))

print('Test accuracy: ',lr.score(X_test_std,y_test))

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue','green','red','cyan','magenta','yellow','black','pink','lightgreen','lightblue','gray','indigo','orange']
weights,params = [], []
for c in np.arange(-4, 6,dtype=float):
    power = 10**c
    lr = LogisticRegression(penalty='l1', C=power, random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)


weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params , weights[:, column],label=df_wine.columns[column+1],color=color)

plt.axhline(0,color='black',linestyle='--',linewidth=3)
plt.xlim([10**(-5),10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38,1.03),ncol=1, fancybox=True)
plt.show()

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0,  n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" %(f+1, 30, feat_labels[f],importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')

plt.xticks(range(X_train.shape[1]),
           feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

X_selected = forest.transform(X_train, threshold=0.15)
print(X_selected.shape)