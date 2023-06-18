from sklearn.linear_model import LinearRegression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import test_pymilo_regression

MODEL_NAME = "Linear-Regression"


def test_linear_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create linear regression object
    linear_regression = LinearRegression()
    # Train the model using the training sets
    linear_regression.fit(x_train, y_train)
    assert test_pymilo_regression(linear_regression, MODEL_NAME, (x_test, y_test)) == True 
