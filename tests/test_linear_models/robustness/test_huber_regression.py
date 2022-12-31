from sklearn.linear_model import HuberRegressor
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Huber-Regressor"


def test_huber_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Huber regression object
    huber_regresion = HuberRegressor()
    # Train the model using the training sets
    huber_regresion.fit(x_train, y_train)
    return test_pymilo_regression(
        huber_regresion, MODEL_NAME, (x_test, y_test))
