from sklearn.linear_model import RidgeClassifierCV
from pymilo.utils.test_pymilo import test_pymilo_classification
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "Ridge-Classifier-CV"


def test_ridge_classifier_cv():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    # Create ridge classifier cv object
    ridge_cv_alphas = [1e-3, 1e-2, 1e-1, 1]
    ridge_classifier = RidgeClassifierCV(alphas=ridge_cv_alphas)
    # Train the model using the training sets
    ridge_classifier.fit(x_train, y_train)
    assert test_pymilo_classification(
        ridge_classifier, MODEL_NAME, (x_test, y_test)) == True 
