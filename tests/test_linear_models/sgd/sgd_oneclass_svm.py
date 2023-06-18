from sklearn.linear_model import SGDOneClassSVM
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test

MODEL_NAME = "SGD-OneClass-Regression"


def sgd_oneclass_svm():
    x_train, _, x_test, y_test = prepare_simple_regression_datasets()
    # Create SGDOneClassSVM regression object
    sgd_random_state = 34
    sgd_oneclass_svm = SGDOneClassSVM(random_state=sgd_random_state)
    # Train the model using the training sets
    sgd_oneclass_svm.fit(x_train)
    assert pymilo_regression_test(
        sgd_oneclass_svm, MODEL_NAME, (x_test, y_test)) == True
