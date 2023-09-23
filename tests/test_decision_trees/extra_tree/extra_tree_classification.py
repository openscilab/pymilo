from sklearn.tree import ExtraTreeClassifier

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "Extra Tree Classifier"

def extra_tree_classification():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    # Create Decision Tree Regressor
    extra_tree_classifier = ExtraTreeClassifier(random_state=1)
    extra_tree_classifier = extra_tree_classifier.fit(x_train, y_train)
    assert pymilo_classification_test(
        extra_tree_classifier, MODEL_NAME, (x_test, y_test)) == True 
