from sklearn.cluster import Birch

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Birch"

def birch():
    x, y = prepare_simple_clustering_datasets()
    birch = Birch().fit(x, y)
    pymilo_clustering_test(birch, MODEL_NAME, True)
