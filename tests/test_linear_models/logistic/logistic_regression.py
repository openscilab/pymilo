from sklearn.linear_model import LogisticRegression
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Logistic-Regression"


def logistic_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Logistic regression object
    logistic_regression_random_state = 4
    logistic_regression = LogisticRegression(
        random_state=logistic_regression_random_state)
    # Train the model using the training sets
    logistic_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        logistic_regression, MODEL_NAME, (x_test, y_test)) == True
