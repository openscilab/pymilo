from sklearn.linear_model import ARDRegression
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Automatic-Relevance-Determination-Regression"


def test_ard_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create ARD regression object
    ard_regression = ARDRegression()
    # Train the model using the training sets
    ard_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        ard_regression, MODEL_NAME, (x_test, y_test)) == True 
