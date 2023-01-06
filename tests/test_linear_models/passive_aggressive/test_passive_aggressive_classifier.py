from sklearn.linear_model import PassiveAggressiveClassifier
from pymilo.utils.test_pymilo import test_pymilo_classification
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "Passive-Aggressive-Classifier"


def test_passive_aggressive_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    # Create ridge regression object
    pac_max_iter = 1000
    pac_random_state = 0
    pac_tol = 1e-3
    passive_aggressive_classifier = PassiveAggressiveClassifier(
        max_iter=pac_max_iter, random_state=pac_random_state, tol=pac_tol)
    # Train the model using the training sets
    passive_aggressive_classifier.fit(x_train, y_train)
    return test_pymilo_classification(
        passive_aggressive_classifier, MODEL_NAME, (x_test, y_test))
