from sklearn.linear_model import MultiTaskLassoCV
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Multi-Task-Lasso-CV-Regression"


def test_multi_task_lasso_cv():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    y_train = [[y, y**2] for y in y_train]
    y_test = [[y, y**2] for y in y_test]
    # Create Multi Task Lasso CV regression object
    lasso_alphas = [1e-3, 1e-2, 1e-1, 1]
    lasso_cv = 5
    lasso_random_state = 0
    multi_task_lasso_cv = MultiTaskLassoCV(
        random_state=lasso_random_state,
        alphas=lasso_alphas,
        cv=lasso_cv)
    # Train the model using the training sets
    multi_task_lasso_cv.fit(x_train, y_train)
    assert pymilo_regression_test(
        multi_task_lasso_cv, MODEL_NAME, (x_test, y_test)) == True 
