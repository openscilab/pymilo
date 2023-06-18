from sklearn.linear_model import LogisticRegressionCV
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_logistic_regression_datasets

MODEL_NAME = "Logistic-Regression-CV"


def test_logistic_regression_cv():
    x_train, y_train, x_test, y_test = prepare_logistic_regression_datasets()
    # Create Logistic regression cv object
    logistic_regression_cv = 5
    logistic_regression_random_state = 0
    logistic_regression_cv = LogisticRegressionCV(
        cv=logistic_regression_cv,
        random_state=logistic_regression_random_state)
    # Train the model using the training sets
    logistic_regression_cv.fit(x_train, y_train)
    assert test_pymilo_regression(
        logistic_regression_cv, MODEL_NAME, (x_test, y_test)) == True 
