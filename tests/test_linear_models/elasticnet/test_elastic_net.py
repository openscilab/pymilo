from sklearn.linear_model import ElasticNet
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Elastic-Net-Regression"


def test_elastic_net():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Elastic Net regression object
    elasticnet_alpha = 0.1
    elasticnet_random_state = 0
    elasticnet_regression = ElasticNet(
        random_state=elasticnet_random_state,
        alpha=elasticnet_alpha)
    # Train the model using the training sets
    elasticnet_regression.fit(x_train, y_train)
    assert test_pymilo_regression(
        elasticnet_regression, MODEL_NAME, (x_test, y_test)) == True 
