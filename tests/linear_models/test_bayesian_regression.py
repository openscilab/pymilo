from  sklearn.linear_model import BayesianRidge
from test_pymilo import test_pymilo_regression
from data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Bayesian-Ridge-Regression"

def test_bayesian_regression():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create bayesian ridge regression object
    bayesian_ridge_regression = BayesianRidge()
    # Train the model using the training sets
    bayesian_ridge_regression.fit(x_train, y_train)
    return test_pymilo_regression(bayesian_ridge_regression, MODEL_NAME, (x_test,y_test))