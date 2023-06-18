from sklearn.linear_model import OrthogonalMatchingPursuitCV
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "Orthogonal-Matching-Pursuit-CV-Regression"


def test_omp_cv():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create Orthogonal Matching Pursuit CV regression object
    omp_cv = 5
    omp_cv_regression = OrthogonalMatchingPursuitCV(cv=omp_cv)
    # Train the model using the training sets
    omp_cv_regression.fit(x_train, y_train)
    assert test_pymilo_regression(
        omp_cv_regression, MODEL_NAME, (x_test, y_test)) == True 
