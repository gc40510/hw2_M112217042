import numpy as np
import matplotlib.pyplot as plt 
from xgboost import XGBRegressor
import pandas as pd  
import seaborn as sns 
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance
from matplotlib import pyplot

data = pd.read_csv('BostonHousing.csv')
#print(data.isnull().sum())

X = data.drop(['medv'], axis = 1)
y = data['medv']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
# define model
model = XGBRegressor()

# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
# evaluate model
#MAPE、RMSE、R^2
rmse = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
rmse = np.absolute(rmse)
print("RMSE:",rmse.mean())
mape = cross_val_score(model, X, y, scoring='neg_mean_absolute_percentage_error', cv=cv, n_jobs=-1)
mape = np.absolute(mape)
print("MAPE:",mape.mean())
r2 = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)
r2 = np.absolute(r2)
print("R^2:",r2.mean())

#------------------------------------------------------------------------------------

xgb = XGBRegressor(n_estimators=100)
xgb.fit(X_train, y_train)

xgb.feature_importances_

# fit model no training data
model = XGBRegressor()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()

#------------------------------------------------------------------------------------

# select features using threshold
thresh=0.01

selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X = selection.transform(X)
# train model
selection_model = XGBRegressor()
selection_model.fit(select_X, y)
# eval model
#MAPE、RMSE、R^2
rmse = cross_val_score(model, select_X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
rmse = np.absolute(rmse)
print("(After select) RMSE:",rmse.mean())
mape = cross_val_score(model, select_X, y, scoring='neg_mean_absolute_percentage_error', cv=cv, n_jobs=-1)
mape = np.absolute(mape)
print("(After select) MAPE:",mape.mean())
r2 = cross_val_score(model, select_X, y, scoring='r2', cv=cv, n_jobs=-1)
r2 = np.absolute(r2)
print("(After select) R^2:",r2.mean())


