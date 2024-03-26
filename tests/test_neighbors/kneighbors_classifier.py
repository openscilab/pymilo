from sklearn.neighbors import KNeighborsClassifier

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "KNeighborsClassifier"

def kneighbors_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    kneighbors_classifier = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    pymilo_classification_test(kneighbors_classifier, MODEL_NAME, (x_test, y_test))
