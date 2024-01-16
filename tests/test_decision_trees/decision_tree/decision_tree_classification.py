from sklearn.tree import DecisionTreeClassifier

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "Decision Tree Classifier"

def decision_tree_classification():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    # Create Decision Tree Regressor
    decision_tree_classifier = DecisionTreeClassifier(random_state=1)
    decision_tree_classifier = decision_tree_classifier.fit(x_train, y_train)
    assert pymilo_classification_test(
        decision_tree_classifier, MODEL_NAME, (x_test, y_test)) == True
