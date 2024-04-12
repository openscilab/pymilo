from sklearn.ensemble import ExtraTreesClassifier
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "ExtraTreesClassifier"

def extra_trees_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    extra_trees_classifier = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(x_train, y_train)
    pymilo_classification_test(extra_trees_classifier, MODEL_NAME, (x_test, y_test))
