from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.util import has_named_parameter

MODEL_NAME = "BaggingRegressor"

def bagging_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    if has_named_parameter(BaggingRegressor, "estimator"):
        bagging_regressor = BaggingRegressor(estimator=SVR(), n_estimators=10, random_state=0).fit(x_train, y_train)
    else:
        bagging_regressor = BaggingRegressor(n_estimators=10, random_state=0).fit(x_train, y_train)
    pymilo_regression_test(bagging_regressor, MODEL_NAME, (x_test, y_test))
