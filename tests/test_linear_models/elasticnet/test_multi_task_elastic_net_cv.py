from sklearn.linear_model import MultiTaskElasticNetCV
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Multi-Task-Elastic-Net-CV-Regression"


def test_multi_task_elastic_net_cv():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    y_train = [[y, y**2] for y in y_train]
    y_test = [[y, y**2] for y in y_test]
    # Create MultiTaskElasticNetCV regression object
    elasticnet_alphas = [1e-3, 1e-2, 1e-1, 1]
    elasticnet_cv = 5
    elasticnet_random_state = 0
    multitask_elasticnet_cv_regression = MultiTaskElasticNetCV(
        random_state=elasticnet_random_state, alphas=elasticnet_alphas, cv=elasticnet_cv)
    # Train the model using the training sets
    multitask_elasticnet_cv_regression.fit(x_train, y_train)
    assert test_pymilo_regression(
        multitask_elasticnet_cv_regression, MODEL_NAME, (x_test, y_test)) == True 
