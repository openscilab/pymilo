from sklearn.neighbors import RadiusNeighborsRegressor

from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "RadiusNeighborsRegressor"

def radius_neighbors_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    radius_neighbors_regressor = RadiusNeighborsRegressor(radius=1.0).fit(x_train, y_train)
    pymilo_regression_test(radius_neighbors_regressor, MODEL_NAME, (x_test, y_test))
