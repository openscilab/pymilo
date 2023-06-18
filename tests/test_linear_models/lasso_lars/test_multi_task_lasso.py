from sklearn.linear_model import MultiTaskLasso
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Multi-Task-Lasso-Regression"


def test_multi_task_lasso():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    y_train = [[y, y**2] for y in y_train]
    y_test = [[y, y**2] for y in y_test]
    # Create MultiTaskLasso regression object
    lasso_alpha = 0.1
    lasso_random_state = 0
    multi_task_lasso = MultiTaskLasso(
        random_state=lasso_random_state,
        alpha=lasso_alpha)
    # Train the model using the training sets
    multi_task_lasso.fit(x_train, y_train)
    assert test_pymilo_regression(
        multi_task_lasso, MODEL_NAME, (x_test, y_test)) == True 
