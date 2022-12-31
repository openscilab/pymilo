from sklearn.linear_model import RidgeClassifier
from pymilo.utils.test_pymilo import test_pymilo_classification
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "Ridge-Classifier"


def test_ridge_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    # Create ridge classifier object
    ridge_alpha = 0.4
    ridge_classifier = RidgeClassifier(alpha=ridge_alpha)
    # Train the model using the training sets
    ridge_classifier.fit(x_train, y_train)
    return test_pymilo_classification(
        ridge_classifier, MODEL_NAME, (x_test, y_test))
