from sklearn.linear_model import PassiveAggressiveRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Passive-Aggressive-Regressor"


def passive_agressive_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Passive Aggressive Regression object
    par_random_state = 2
    par_max_iter = 100
    passive_aggressive_regression = PassiveAggressiveRegressor(
        max_iter=par_max_iter, random_state=par_random_state)
    # Train the model using the training sets
    passive_aggressive_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        passive_aggressive_regression, MODEL_NAME, (x_test, y_test)) == True
