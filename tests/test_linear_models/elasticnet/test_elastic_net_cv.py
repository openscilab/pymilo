from sklearn.linear_model import ElasticNetCV
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Elastic-Net-CV-Regression"


def test_elastic_net_cv():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Elastic Net CV regression object
    elasticnet_alphas = [1e-3, 1e-2, 1e-1, 1]
    elasticnet_cv = 5
    elasticnet_random_state = 0
    elasticnet_cv_regression = ElasticNetCV(
        cv=elasticnet_cv,
        alphas=elasticnet_alphas,
        random_state=elasticnet_random_state)
    # Train the model using the training sets
    elasticnet_cv_regression.fit(x_train, y_train)
    return test_pymilo_regression(
        elasticnet_cv_regression, MODEL_NAME, (x_test, y_test))
