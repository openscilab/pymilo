from sklearn.ensemble import RandomForestClassifier
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "RandomForestClassifier"

def random_forest_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    random_forest_classifier = RandomForestClassifier(max_depth=2, random_state=0).fit(x_train, y_train)
    pymilo_classification_test(random_forest_classifier, MODEL_NAME, (x_test, y_test))
