from sklearn.linear_model import HuberRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Huber-Regressor"


def huber_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Huber regression object
    huber_regresion = HuberRegressor(max_iter=300)
    # Train the model using the training sets
    huber_regresion.fit(x_train, y_train)
    assert pymilo_regression_test(
        huber_regresion, MODEL_NAME, (x_test, y_test)) == True
