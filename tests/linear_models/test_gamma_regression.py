from  sklearn.linear_model import GammaRegressor
from test_pymilo import test_pymilo_regression
from data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Gamma-Regression"

def test_gamma_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Gamma regression object
    gamma_alpha = 0.5
    gamma_regression = GammaRegressor(alpha= gamma_alpha)
    # Train the model using the training sets
    gamma_regression.fit(x_train, y_train)
    return test_pymilo_regression(gamma_regression, MODEL_NAME, (x_test,y_test))