from sklearn.linear_model import TweedieRegressor
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test

MODEL_NAME = "Tweedie-Regression"


def test_tweedie_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Tweedie Regression object
    tweedie_alpha = 0.5
    tweedie_link = 'log'
    tweedie_power = 1
    tweedie_regression = TweedieRegressor(
        power=tweedie_power,
        alpha=tweedie_alpha,
        link=tweedie_link)
    # Train the model using the training sets
    tweedie_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        tweedie_regression, MODEL_NAME, (x_test, y_test)) == True 
