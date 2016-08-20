from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import neighbors
import numpy as np 
import pandas as pd

#dataframe

df = pd.read_csv('stock_returns_base150.csv', skipinitialspace=True)
df = df[0:100]

df['newDates'] =  pd.to_datetime([str(val).strip(' 0:00') for val in df['date'] if val is not np.nan])
df = df.drop('date',axis=1)

y = df.loc[0:49,['newDates','S1']]
X = df.loc[0:49,df.columns[1:]]

X = X.set_index(['newDates'])
y = y.set_index(['newDates'])

X = np.asmatrix(X)
yy = y
y = np.asarray(y.values.ravel(),dtype="|S6")

toPredict = df.loc[50:100,df.columns[1:]]
toPredict = toPredict.set_index(['newDates'])
toPredict = np.asmatrix(toPredict)

results = []
clf = None
winner = None
#pca
for nComp in range(2,10):
	pca = PCA(n_components=nComp)
	X_pca 		  = np.asmatrix(pca.fit_transform(X))
	toPredict_pca = np.asmatrix(pca.fit_transform(toPredict)) # pca on prediction dataset


	# algo 1 (knn)
	if clf is not None: del clf
	parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9],'weights':('uniform','distance'), 'algorithm': ['auto','ball_tree', 'kd_tree', 'brute'], 'leaf_size': [r for r in range(1,51) if r % 5 is 0],'p':[1,2]} # 'weights':('uniform','distance')
	neigh = neighbors.KNeighborsRegressor(n_jobs=-1)
	clf = grid_search.GridSearchCV(neigh, parameters,cv=5,n_jobs=-1)
	clf.fit(X_pca , np.array(y).astype(np.float))
	if winner is None: 
		winner =  clf
	elif clf.best_score_ > winner.best_score_:
		del winner
		winner = clf
		prediction_winner = toPredict_pca

	# # print algo 1 results
	# print '\n', clf.best_estimator_
	# print '\n', clf.best_score_
	results.append([ pca.n_components, clf.best_estimator_, clf.best_score_])

	# algo 2 (linear regression)
	del clf
	parameters = {'fit_intercept': [True, False], 'normalize':[True, False],'copy_X':[True, False] }
	regress = linear_model.LinearRegression(n_jobs=-1)
	clf = grid_search.GridSearchCV(regress, parameters,cv=5)
	clf.fit(X_pca , np.array(y).astype(np.float))
	if winner is None: 
		winner =  clf
	elif clf.best_score_ > winner.best_score_:
		del winner
		winner = clf
		prediction_winner = toPredict_pca

	# #print algo 2 results
	# print 'Linear Regression Results: '
	# print '\n', clf.best_estimator_
	# print '\n', clf.best_score_

	results.append([ pca.n_components, clf.best_estimator_, clf.best_score_])

print '\n\nBest test: '

bestScore = 0; bestTest = 0;
for test in results:
	if test[2] > bestScore:
		bestScore = test[2]
		bestTest = test
print bestTest

print winner.predict(prediction_winner)