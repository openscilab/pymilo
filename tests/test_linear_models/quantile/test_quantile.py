from sklearn.linear_model import QuantileRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Quantile-Regressor"


def test_quantile_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Quantile regression object
    quantile_regression = QuantileRegressor(quantile=0.8, solver="highs")
    # Train the model using the training sets
    quantile_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        quantile_regression, MODEL_NAME, (x_test, y_test)) == True 
