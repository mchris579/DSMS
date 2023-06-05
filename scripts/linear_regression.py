import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as ps
from sklearn.compose import TransformedTargetRegressor
import numpy as np


joblib_file = "models/linear_regression/lr.pkl"
x_train = ps.read_csv('data/split/x_train.csv', engine='python', sep=',')
y_train = ps.read_csv('data/split/y_train.csv', engine='python', sep=',')

model = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log, inverse_func=np.exp)
model.fit(x_train, y_train)

joblib.dump(model, joblib_file)
