from sklearn.ensemble import RandomForestRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "RandomForestRegressor"

def random_forest_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    random_forest_regressor = RandomForestRegressor(max_depth=2, random_state=0).fit(x_train, y_train, sample_weight=1)
    pymilo_regression_test(random_forest_regressor, MODEL_NAME, (x_test, y_test))
