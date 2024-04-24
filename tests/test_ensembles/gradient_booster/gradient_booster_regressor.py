from sklearn.ensemble import GradientBoostingRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "GradientBoostingRegressor"

def gradient_booster_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    gradient_booster_regressor = GradientBoostingRegressor(random_state=0).fit(x_train, y_train, sample_weight=1)
    pymilo_regression_test(gradient_booster_regressor, MODEL_NAME, (x_test, y_test))
