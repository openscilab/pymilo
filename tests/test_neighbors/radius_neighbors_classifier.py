from sklearn.neighbors import RadiusNeighborsClassifier

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "RadiusNeighborsClassifier"

def radius_neighbors_classifier():
    x_train, y_train, _, _ = prepare_simple_classification_datasets()
    radius_neighbors_classifier = RadiusNeighborsClassifier(radius=1.0).fit(x_train, y_train)
    pymilo_classification_test(radius_neighbors_classifier, MODEL_NAME, (x_train, y_train))
