from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as ps
import numpy as np
import joblib
from random import randint


joblib_file = "models/random_forest/rf.pkl"
joblib_file_hyper = "models/random_forest/rf_tuning.pkl"
x_train = ps.read_csv('data/split/x_train.csv', engine='python', sep=',')
y_train = ps.read_csv('data/split/y_train.csv', engine='python', sep=',')


rf = RandomForestClassifier()
model = rf.fit(x_train, y_train.values.ravel())

joblib.dump(model, joblib_file)

# RandomForest with hyperparameter tuning
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth}


rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search object to the data
model_hyper = rand_search.fit(x_train, y_train.values.ravel())

model_hyper.best_params_

joblib.dump(model_hyper, joblib_file_hyper)






