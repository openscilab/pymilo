from sklearn.neighbors import KNeighborsRegressor

from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "KNeighborsRegressor"

def kneighbors_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    kneighbors_regressor = KNeighborsRegressor(n_neighbors=2).fit(x_train, y_train)
    pymilo_regression_test(kneighbors_regressor, MODEL_NAME, (x_test, y_test))
