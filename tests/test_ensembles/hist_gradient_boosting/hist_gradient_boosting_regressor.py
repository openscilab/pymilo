from sklearn.ensemble import HistGradientBoostingRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "HistGradientBoostingRegressor"

def hist_gradient_boosting_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    hist_gradient_boosting_regressor = HistGradientBoostingRegressor().fit(x_train, y_train)
    pymilo_regression_test(hist_gradient_boosting_regressor, MODEL_NAME, (x_test, y_test))

