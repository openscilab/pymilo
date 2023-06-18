from sklearn.linear_model import SGDRegressor
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test

MODEL_NAME = "SGD-Regression"


def sgd_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create SGD Regression object
    sgd_max_iter = 100000
    sgd_tol = 1e-3
    sgd_regression = SGDRegressor(max_iter=sgd_max_iter, tol=sgd_tol)
    # Train the model using the training sets
    sgd_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        sgd_regression, MODEL_NAME, (x_test, y_test)) == True 
