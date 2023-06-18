from sklearn.linear_model import LassoLars
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Lasso-Lars-Regression"


def test_lasso_lars():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Lasso Lars regression object
    lasso_alpha = 0.2
    lasso_lars_regression = LassoLars(lasso_alpha)
    # Train the model using the training sets
    lasso_lars_regression.fit(x_train, y_train)
    assert test_pymilo_regression(
        lasso_lars_regression, MODEL_NAME, (x_test, y_test)) == True 
