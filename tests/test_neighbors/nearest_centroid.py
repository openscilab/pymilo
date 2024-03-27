from sklearn.neighbors import NearestCentroid

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "NearestCentroid"

def nearest_centroid():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    nearest_centroid = NearestCentroid().fit(x_train, y_train)
    pymilo_classification_test(nearest_centroid, MODEL_NAME, (x_test, y_test))
