from sklearn.ensemble import RandomForestClassifier
mport pandas as pd
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
feat_labels = def.wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0m n_jobs=-1)
forest.fit(X_train, y_train)