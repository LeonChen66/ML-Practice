import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',header=None, sep='\s+')
# df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
# df.to_csv('house.csv')
df = pd.read_csv('house.csv')
# sns.set(style='whitegrid',context='notebook')
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
# sns.pairplot(df[cols],size=1.5)
# plt.show()
# cm = np.corrcoef(df[cols].values.T)
# print(cm)
# sns.set(font_scale=1.5)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
# plt.show()

X = df[['RM']].values
y = df['MEDV'].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
slr = LinearRegression()
slr.fit(X,y)
print('Slope:%f' %slr.coef_[0])
def lin_regplot(X,y,model):
    plt.scatter(X,y,c='blue')
    plt.plot(X,model.predict(X),color='red')
    return None

# lin_regplot(X,y,slr)
# plt.show()

ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         residual_metric=lambda x:np.sum(np.abs(x),axis=1),
                         residual_threshold=5.0,
                         random_state=0)

ransac.fit(X,y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3,10,1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.figure(1)
plt.scatter(X[inlier_mask],y[inlier_mask],c='blue',marker='o',label='Inliers')
plt.scatter(X[outlier_mask],y[outlier_mask],c='lightgreen',marker='s',label='Outliers')
plt.plot(line_X,line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\' s [MEDV] ')

# plt.show()

print('Slope: %f' %ransac.estimator_.coef_[0])

X = df.iloc[:,:-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
plt.figure(2)
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue',marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')

# plt.show()

print('MES train %f, test:%f' %(mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test,y_test_pred)))

X = df[['LSTAT']].values
y = df['MEDV'].values
regr = LinearRegression()
#create polynomial features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)
#linear fit
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X,y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))
#quadratic fit
regr = regr.fit(X_quad,y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))
#cubic fit
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))
#plot
plt.figure(3)
plt.scatter(X,y,label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1) , $R^2=%.2f$' %linear_r2, color='blue',lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label='linear (d=2) , $R^2=%.2f$' %quadratic_r2, color='red',lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label='linear (d=3) , $R^2=%.2f$' %cubic_r2, color='green',lw=2, linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')

# plt.show()

tree =  DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
plt.figure(4)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx],y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')

X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
forest = RandomForestRegressor(n_estimators=1000,criterion='mse',random_state=1, n_jobs=1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print('MSE train: %f, test: %f'%(mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test,y_test_pred)))
print('R^2 train: %f, test: %f'%(r2_score(y_train, y_train_pred),r2_score(y_test,y_test_pred)))
plt.figure(5)
plt.scatter(y_train_pred,y_train_pred-y_train,c='black',marker='o',s=35,alpha=0.5,label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='lightgreen',marker='s',s=35,alpha=0.7,label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()