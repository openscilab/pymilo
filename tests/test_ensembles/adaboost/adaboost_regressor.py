from sklearn.ensemble import AdaBoostRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "AdaBoostRegressor"

def adaboost_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    adaboost_regressor = AdaBoostRegressor(random_state=0, n_estimators=100).fit(x_train, y_train)
    pymilo_regression_test(adaboost_regressor, MODEL_NAME, (x_test, y_test))
