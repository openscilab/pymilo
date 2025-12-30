import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.compose import TransformedTargetRegressor
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test

MODEL_NAME = "TransformedTargetRegressor"


def transformed_target_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()

    tt_regressor = TransformedTargetRegressor(regressor=LinearRegression(),
                                func=np.log, inverse_func=np.exp)

    tt_regressor.fit(x_train, y_train)

    assert pymilo_regression_test(
        tt_regressor, MODEL_NAME, (x_test, y_test)) == True


def complex_transformed_target_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create SGD Regression object
    sgd_max_iter = 1000
    sgd_tol = 1e-3
    sgd_regression = SGDRegressor(max_iter=sgd_max_iter, tol=sgd_tol)
    # Train the model using the training sets
    sgd_regression.fit(x_train, y_train)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    ttr = TransformedTargetRegressor(regressor=sgd_regression, transformer=y_scaler)

    pipeline = Pipeline([("x_scaler", x_scaler), ("ttr", ttr)])
    pipeline.fit(x_train, y_train)

    assert pymilo_regression_test(
        pipeline, MODEL_NAME, (x_test, y_test)) == True
