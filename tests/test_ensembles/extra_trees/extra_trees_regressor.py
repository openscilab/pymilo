from sklearn.ensemble import ExtraTreesRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "ExtraTreesRegressor"

def extra_trees_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    extra_trees_regressor = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(x_train, y_train)
    pymilo_regression_test(extra_trees_regressor, MODEL_NAME, (x_test, y_test))
