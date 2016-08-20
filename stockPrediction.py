from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import random
import pandas as pd

df = pd.read_csv('stock_returns_base150.csv', skipinitialspace=True)
df = df[0:100]

df['newDates'] =  pd.to_datetime([str(val).strip(' 0:00') for val in df['date'] if val is not np.nan])
df = df.drop('date',axis=1)

y = df.loc[0:49,['newDates','S1']]
X = df.loc[0:49,df.columns[1:]]

X = X.set_index(['newDates'])
y = y.set_index(['newDates'])

X = np.asmatrix(X)
y = np.asarray(y,dtype="|S6")

## PCA
#for nComp in range(2,10):
nComp = 7
pca = PCA(n_components=nComp)
X_pca = np.asmatrix(pca.fit_transform(X))

print 'n = {}, explained variance for PCA = {}'.format(pca.n_components, pca.explained_variance_ratio_)

## ensemble 
#linear regression or general linear model, logistical regression, and k nearest neighbors

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = KNeighborsClassifier(n_neighbors=7)
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('knn', clf3)], voting='soft')

params = {'lr__C': [1.0, 100.0]} # 'rf__n_estimators': [20, 200]

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=4)
print X_pca.shape
grid = grid.fit(X_pca, y)

# print results
