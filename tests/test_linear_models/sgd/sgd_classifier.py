from sklearn.linear_model import SGDClassifier
from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from pymilo.utils.test_pymilo import pymilo_classification_test
MODEL_NAME = "SGD-Classifier"


def sgd_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    # Create SGDClassifier regression object
    sgd_max_iter = 100000
    sgd_tol = 1e-3
    sgd_classifier = SGDClassifier(max_iter=sgd_max_iter, tol=sgd_tol)
    # Train the model using the training sets
    sgd_classifier.fit(x_train, y_train)
    assert pymilo_classification_test(
        sgd_classifier, MODEL_NAME, (x_test, y_test)) == True 
