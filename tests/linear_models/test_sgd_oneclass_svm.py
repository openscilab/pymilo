from  sklearn.linear_model import SGDOneClassSVM
from data_exporter import prepare_simple_regression_datasets
from test_pymilo import test_pymilo_regression

MODEL_NAME = "SGD-OneClass-Regression"

def test_sgd_oneclass_svm():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create SGDOneClassSVM regression object
    sgd_random_state = 34
    sgd_oneclass_svm = SGDOneClassSVM(random_state= sgd_random_state)
    # Train the model using the training sets
    sgd_oneclass_svm.fit(x_train)
    return test_pymilo_regression(sgd_oneclass_svm, MODEL_NAME, (x_test,y_test))
