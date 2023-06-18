from sklearn.linear_model import Ridge
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test
MODEL_NAME = "Ridge-Regression"


def ridge_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create ridge regression object
    ridge_alpha = 0.5
    ridge_regression = Ridge(alpha=ridge_alpha)
    # Train the model using the training sets
    ridge_regression.fit(x_train, y_train)
    assert pymilo_regression_test(
        ridge_regression, MODEL_NAME, (x_test, y_test)) == True 
