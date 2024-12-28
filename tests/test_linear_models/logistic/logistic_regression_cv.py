from sklearn.linear_model import LogisticRegressionCV
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "Logistic-Regression-CV"


def logistic_regression_cv():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    # Create Logistic regression cv object
    logistic_regression_cv = 5
    logistic_regression_random_state = 0
    logistic_regression_cv = LogisticRegressionCV(
        cv=logistic_regression_cv,
        random_state=logistic_regression_random_state)
    # Train the model using the training sets
    logistic_regression_cv.fit(x_train, y_train)
    assert pymilo_classification_test(
        logistic_regression_cv, MODEL_NAME, (x_test, y_test)) == True
