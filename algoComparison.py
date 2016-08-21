from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import neighbors
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

#dataframe creation and preprocessing of data

df = pd.read_csv('stock_returns_base150.csv', skipinitialspace=True)
df = df[0:100]

df['newDates'] =  pd.to_datetime([str(val).strip(' 0:00') for val in df['date'] if val is not np.nan])
df = df.drop('date',axis=1)

y = df.loc[0:49,['newDates','S1']]
X = df.loc[0:49,df.columns[1:]]

# indexes are properly set to dates so not to interfere with the coder and processing 

X = X.set_index(['newDates'])
y = y.set_index(['newDates'])

#conversions that avoid errors
X = np.asmatrix(X)
yy = y
y = np.asarray(y.values.ravel(),dtype="|S6")

# prediction array design

toPredict = df.loc[50:100,df.columns[1:]]
toPredict = toPredict.set_index(['newDates'])
toPredict = np.asmatrix(toPredict)

results = []
clf = None
winner = None
#pca
for nComp in range(2,10): # iterates through each possible component 
	pca = PCA(n_components=nComp)
	X_pca 		  = np.asmatrix(pca.fit_transform(X))
	if nComp is 9: 
		totalVariance = pca.explained_variance_ratio_
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
	
modelPrediction = winner.predict(prediction_winner)

## Mean Squared error prediction from scratch


df = pd.read_csv('stock_returns_base150.csv', skipinitialspace=True)
df = df[0:100]
df['newDates'] =  pd.to_datetime([str(val).strip(' 0:00') for val in df['date'] if val is not np.nan])
df = df.drop('date',axis=1)

X = df.set_index(['newDates'])
X = X.loc[:,X.columns[1:]]
X = X.sum(axis=1)/X.shape[1]
naivePrediction = X[48:98].values
# a is the prediction and the same as variable modelPrediction. They will be interchangable
a = [0.37770974,-0.73307011,0.88773289,0.13026491,-1.86906431,1.52028684,0.15742395,-0.38432996,0.59800504,1.4368041,-2.76458525,0.48187279,0.81824333,-1.15907964,-0.14791345,2.60427284,-0.55643502,1.17828395,0.85551937,1.11863532,-0.95189415,1.68381162,-1.2285826,0.20917567,0.21139037,0.59982421,0.04832726,-1.04204633,0.53017159,-0.20220081,-1.9821128,0.56913762,0.60732677,-0.7248571,-0.52386703,0.01656557,0.79900966,-0.03771517,0.97580942,-2.23002944,2.20280162,-0.83724361,0.02633449,0.3757501,0.40947482,-1.05293429,0.16270305,0.79781595,0.45513208,1.51234413]
#print len(a)
#print 'accuracy = ',accuracy_score(naivePrediction, a)
print 'mean_squared_error = ',mean_squared_error(naivePrediction, modelPrediction)

print 'total variance for pca is :', totalVariance