from sklearn.neighbors import NearestNeighbors

from pymilo.utils.test_pymilo import pymilo_nearest_neighbor_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "NearestNeighbors"

def nearest_neighbor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    nearest_neighbor = NearestNeighbors(n_neighbors=2, radius=0.4).fit(x_train, y_train)
    pymilo_nearest_neighbor_test(nearest_neighbor, MODEL_NAME, (x_test, y_test))
