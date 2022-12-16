from  sklearn.linear_model import Lasso
from test_pymilo import test_pymilo_regression
from data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Lasso-Regression"

def test_lasso():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Lasso regression object
    lasso_alpha = 0.2
    lasso_regression = Lasso(lasso_alpha)
    # Train the model using the training sets
    lasso_regression.fit(x_train, y_train)
    return test_pymilo_regression(lasso_regression, MODEL_NAME, (x_test,y_test))
