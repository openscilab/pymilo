from sklearn.cluster import DBSCAN

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "DBSCAN"

def dbscan():
    x, y = prepare_simple_clustering_datasets()
    dbscan = DBSCAN(eps=3, min_samples=2).fit(x, y)
    pymilo_clustering_test(dbscan, MODEL_NAME, x)
