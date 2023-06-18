from sklearn.linear_model import MultiTaskElasticNet
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Multi-Task-Elastic-Net-Regression"


def test_multi_task_elastic_net():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    y_train = [[y, y**2] for y in y_train]
    y_test = [[y, y**2] for y in y_test]
    # Create MultiTaskElasticNet regression object
    elasticnet_alpha = 0.01
    elasticnet_random_state = 0
    multitask_elasticnet_regression = MultiTaskElasticNet(
        random_state=elasticnet_random_state, alpha=elasticnet_alpha)
    # Train the model using the training sets
    multitask_elasticnet_regression.fit(x_train, y_train)
    assert test_pymilo_regression(
        multitask_elasticnet_regression, MODEL_NAME, (x_test, y_test)) == True 
