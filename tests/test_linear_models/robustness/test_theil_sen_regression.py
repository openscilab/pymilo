from sklearn.linear_model import TheilSenRegressor
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test

MODEL_NAME = "Theil-Sen-Regressor"


def test_theil_sen_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create TheilSen Regression object
    theilsen_random_state = 4
    theilsen_regresion = TheilSenRegressor(random_state=theilsen_random_state)
    # Train the model using the training sets
    theilsen_regresion.fit(x_train, y_train)
    assert pymilo_regression_test(
        theilsen_regresion, MODEL_NAME, (x_test, y_test)) == True 
