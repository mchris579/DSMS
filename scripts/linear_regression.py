import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as ps

joblib_file = "models/linear_regression/lr.pkl"
x_train = ps.read_csv('data/split/x_train.csv', engine='python', sep=',')
y_train = ps.read_csv('data/split/y_train.csv', engine='python', sep=',')


model = LinearRegression().fit(x_train, y_train) 

joblib.dump(model, joblib_file)

