from sklearn.svm import SVR
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as ps

joblib_file = "models/support_vector_regression/svr.pkl"

x_train = ps.read_csv('data/split/x_train.csv', engine='python', sep=',')
y_train = ps.read_csv('data/split/y_train.csv', engine='python', sep=',')

regressor = SVR(kernel='linear')
model = regressor.fit(x_train,y_train.values.ravel())

joblib.dump(model, joblib_file)
