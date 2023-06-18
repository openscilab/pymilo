from sklearn.linear_model import ARDRegression
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Automatic-Relevance-Determination-Regression"


def test_ard_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create ARD regression object
    ard_regression = ARDRegression()
    # Train the model using the training sets
    ard_regression.fit(x_train, y_train)
    assert test_pymilo_regression(
        ard_regression, MODEL_NAME, (x_test, y_test)) == True 
